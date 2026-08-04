"""Recurrent Transformer with Attention Residuals (AttnRes).
Architecture: 256 hidden, 8 unique layers (2 recurrent in the middle, looped 2x),
4 heads, 512 FFN, 64 context, 2048 vocab, 1.58-bit ternary weights.

- Depth-Recurrent: the 2 middle layers reuse the same weights and loop 2 times,
  giving effective depth 3 + 2*2 + 3 = 10 without extra parameters.
- Attention Residuals: each layer uses a learned pseudo-query to attend over all
  preceding layer outputs (attention over depth) instead of a fixed +1 residual.
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
    num_layers: int = 8          # total unique layers
    num_recurrent_layers: int = 2  # middle layers that are looped
    recurrent_loops: int = 2     # how many times the recurrent layers loop
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
        q, k, v = qkv[0], qkv[1], qkv[2]

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
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
    """A single transformer block (attention + MLP) with pre-norm."""
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


class AttentionResidual(nn.Module):
    """Attention over depth: a learned pseudo-query attends over all previous
    layer outputs to produce a selective residual blend instead of a fixed +1.

    prev_outputs: list of (B, T, C) tensors from all preceding layers.
    Returns a blended (B, T, C) residual that replaces the fixed identity skip.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        # Learned pseudo-query (per token position, shared across batch)
        self.query = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.02)
        # Projection for keys/values from each previous layer output
        self.k_proj = QuantLinear(cfg.hidden_size, cfg.hidden_size)
        self.v_proj = QuantLinear(cfg.hidden_size, cfg.hidden_size)
        self.out_proj = QuantLinear(cfg.hidden_size, cfg.hidden_size)
        self.scale = cfg.hidden_size ** -0.5

    def forward(self, prev_outputs):
        if not prev_outputs:
            return 0.0
        # Stack previous layer outputs along a new "depth" dimension
        # prev: (L, B, T, C)
        prev = torch.stack(prev_outputs, dim=0)
        L, B, T, C = prev.shape

        # Keys/values: (B, T, L, C)
        k = self.k_proj(prev).permute(1, 2, 0, 3)
        v = self.v_proj(prev).permute(1, 2, 0, 3)

        # Query: (B, T, 1, C) broadcast over depth
        q = self.query.expand(B, T, 1, C)

        # Attention over depth: (B, T, 1, L)
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = F.softmax(att, dim=-1)

        # Blend: (B, T, 1, C) -> (B, T, C)
        out = (att @ v).squeeze(2)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Full Recurrent Transformer with AttnRes
# ---------------------------------------------------------------------------
class RecurrentTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.wpe = nn.Embedding(cfg.max_seq_len, cfg.hidden_size)
        self.drop = nn.Dropout(cfg.dropout)

        # Split layers: pre | recurrent (shared) | post
        n_pre = (cfg.num_layers - cfg.num_recurrent_layers) // 2
        n_post = cfg.num_layers - n_pre - cfg.num_recurrent_layers

        self.pre_layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(n_pre)])
        # Recurrent layers: shared weights, looped `recurrent_loops` times
        self.recurrent_layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.num_recurrent_layers)]
        )
        self.post_layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(n_post)])

        # Attention Residuals: one per unique layer (pre + recurrent + post)
        n_unique = n_pre + cfg.num_recurrent_layers + n_post
        self.attn_residuals = nn.ModuleList(
            [AttentionResidual(cfg) for _ in range(n_unique)]
        )

        self.ln_f = RMSNorm(cfg.hidden_size)
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

        # Track all layer outputs for Attention Residuals
        layer_outputs = [x]  # embedding output is the first "layer"

        # Pre layers (unique)
        for i, block in enumerate(self.pre_layers):
            block_out = block(x)
            residual = self.attn_residuals[i](layer_outputs)
            x = block_out + residual
            layer_outputs.append(x)

        # Recurrent layers (shared weights, looped)
        rec_start = len(self.pre_layers)
        for j, block in enumerate(self.recurrent_layers):
            for _ in range(self.cfg.recurrent_loops):
                block_out = block(x)
                residual = self.attn_residuals[rec_start + j](layer_outputs)
                x = block_out + residual
                layer_outputs.append(x)

        # Post layers (unique)
        post_start = rec_start + len(self.recurrent_layers)
        for k, block in enumerate(self.post_layers):
            block_out = block(x)
            residual = self.attn_residuals[post_start + k](layer_outputs)
            x = block_out + residual
            layer_outputs.append(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, tokenizer, prompt, max_new_tokens=64, temperature=1.0, top_k=None):
        """Autoregressive generation capped to max_seq_len."""
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
            if next_id == self.cfg.bos_id:
                break

        self.train()
        return tokenizer.decode(ids)