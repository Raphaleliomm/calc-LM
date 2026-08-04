"""BitNet b1.58-style transformer (1.58-bit ternary weights).
Architecture: 256 hidden, 8 layers, 4 heads, 512 FFN, 64 context, 2048 vocab.
All attention/FFN linear layers use ternary weights {-1, 0, +1} scaled by absmean.
"""

import json
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1.58-bit ternary quantization with Straight-Through Estimator (STE)
# ---------------------------------------------------------------------------
class TernaryQuant(torch.autograd.Function):
    """Quantize weights to {-alpha, 0, +alpha} where alpha = absmean(W).
    Forward uses quantized weights; backward passes gradients through untouched (STE)."""

    @staticmethod
    def forward(ctx, weight):
        alpha = weight.abs().mean().clamp(min=1e-8)
        w_scaled = weight / alpha
        wq = torch.clamp(torch.round(w_scaled), -1, 1)
        return wq * alpha

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def quantize_ternary(w):
    return TernaryQuant.apply(w)


class QuantLinear(nn.Module):
    """Linear layer with 1.58-bit (ternary) quantized weights in the forward pass."""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        wq = quantize_ternary(self.weight)
        return F.linear(x, wq, self.bias)


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    vocab_size: int = 2048
    hidden_size: int = 256
    num_layers: int = 8
    num_heads: int = 4
    ffn_size: int = 512
    max_seq_len: int = 64
    dropout: float = 0.0
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3

    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))

    def to_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.hidden_size % cfg.num_heads == 0
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads

        self.qkv = QuantLinear(cfg.hidden_size, 3 * cfg.hidden_size)
        self.out = QuantLinear(cfg.hidden_size, cfg.hidden_size)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        mask = torch.tril(torch.ones(1, 1, cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, T, D)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # (B, H, T, D)
        y = y.transpose(1, 2).reshape(B, T, C)
        y = self.resid_drop(self.out(y))
        return y


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up = QuantLinear(cfg.hidden_size, cfg.ffn_size)
        self.down = QuantLinear(cfg.ffn_size, cfg.hidden_size)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.gelu(self.up(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class BitNetTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.wpe = nn.Embedding(cfg.max_seq_len, cfg.hidden_size)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])

        self.ln_f = RMSNorm(cfg.hidden_size)
        # Output head stays full precision for stable logits (common practice in BitNet).
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"Sequence length {T} exceeds max {self.cfg.max_seq_len}"

        pos = torch.arange(0, T, device=idx.device, dtype=torch.long)
        x = self.wte(idx) + self.wpe(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, tokenizer, prompt, max_new_tokens=64, temperature=1.0, top_k=None, do_sample=True):
        """Autoregressive generation. max_new_tokens capped so total length <= max_seq_len."""
        self.eval()
        device = next(self.parameters()).device

        ids = tokenizer.encode(prompt).ids
        if not ids:
            ids = [self.cfg.bos_id]
        if len(ids) > self.cfg.max_seq_len:
            ids = ids[-self.cfg.max_seq_len:]

        max_new = min(max_new_tokens, self.cfg.max_seq_len - len(ids))
        if max_new <= 0:
            return tokenizer.decode(ids)[: len(prompt)] + ""

        for _ in range(max_new):
            x = torch.tensor(ids[-self.cfg.max_seq_len:], dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = self(x)
            logits = logits[0, -1, :]
            if temperature <= 0:
                next_id = int(logits.argmax().item())
            else:
                logits = logits / temperature
                if top_k is not None:
                    topv, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < topv[-1]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
            ids.append(next_id)
            if next_id == self.cfg.bos_id:  # stop signal placeholder (bos won't normally be generated)
                break

        self.train()
        return tokenizer.decode(ids)
