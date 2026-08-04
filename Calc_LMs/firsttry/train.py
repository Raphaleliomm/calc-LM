"""Train the BitNet b1.58 transformer (no-world-knowledge) on chat data."""

import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

from model import BitNetTransformer, ModelConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.environ.get(
    "TRAIN_DATA",
    os.path.join(
        BASE_DIR,
        "..",
        "..",
        "kaggle_output",
        "world_knowledge_classification",
        "rank0_no_world_knowledge_confidence_5_plus.jsonl",
    ),
)
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "checkpoints"))

MAX_SEQ_LEN = 64
BATCH_SIZE = 8
GRAD_ACCUM = 4          # effective batch = 8 * 4 = 32
LEARNING_RATE = 3e-3
MIN_LR = 1e-4
WARMUP_STEPS = 200
MAX_STEPS = int(os.environ.get("TRAIN_STEPS", "5000"))
LOG_EVERY = 25
EVAL_EVERY = 250
SAVE_EVERY = 500
SEED = 42
MAX_GRAD_NORM = 1.0
DROPOUT = 0.0
VAL_RATIO = 0.02        # small holdout for eval

torch.manual_seed(SEED)
random.seed(SEED)
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset: read JSONL, build (text -> token ids) chat examples
# ---------------------------------------------------------------------------
def load_examples(tokenizer, path):
    """Return list of per-message id lists: {"ids": [...], "mask": [...]}."""
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                sample = json.loads(record["sample"])
                messages = sample.get("messages", [])
            except Exception:
                continue

            ids = []
            mask = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if not content:
                    continue
                enc = tokenizer.encode(content)
                if role == "user":
                    # <bos> + user tokens (masked out, not predicted)
                    ids.extend([bos_id] + enc.ids)
                    mask.extend([-100] + [-100] * len(enc.ids))
                elif role == "assistant":
                    # assistant tokens + <eos> (predicted for LM objective)
                    ids.extend(enc.ids + [eos_id])
                    mask.extend([1] * len(enc.ids) + [1])
                else:
                    ids.extend(enc.ids)
                    mask.extend([-100] * len(enc.ids))

            examples.append({"ids": ids, "mask": mask})
    return examples


class ChatDataset(Dataset):
    """Causal LM dataset with shifted targets: input[t] -> target[t+1].
    Only assistant tokens (and their trailing <eos>) are trained on."""

    def __init__(self, tokenizer, examples, max_len=64, rng=None):
        self.max_len = max_len
        self.rng = rng or random.Random(0)
        self.pad_id = tokenizer.token_to_id("<pad>")

        self.sequences = []   # (input_ids, target_ids)
        for ex in examples:
            ids = ex["ids"]
            tok_mask = ex["mask"]   # -100 = ignore, 1 = train

            # chunk into max_len blocks with target shift by +1
            for start in range(0, len(ids), max_len):
                in_ids = ids[start:start + max_len]
                tgt_ids = ids[start + 1:start + max_len + 1]
                tgt_mask = tok_mask[start + 1:start + max_len + 1]
                if not tgt_ids:
                    continue
                self.sequences.append((in_ids, tgt_ids, tgt_mask))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        in_ids, tgt_ids, tgt_mask = self.sequences[idx]
        pad = self.pad_id
        # Pad both input and target independently to max_len.
        if len(in_ids) < self.max_len:
            in_ids = in_ids + [pad] * (self.max_len - len(in_ids))
        if len(tgt_ids) < self.max_len:
            tgt_ids = tgt_ids + [-100] * (self.max_len - len(tgt_ids))
            tgt_mask = tgt_mask + [-100] * (self.max_len - len(tgt_mask))
        xs = torch.tensor(in_ids, dtype=torch.long)
        # fold mask into targets: ignored positions become -100
        ys = torch.tensor(tgt_ids, dtype=torch.long)
        m = torch.tensor(tgt_mask, dtype=torch.long)
        ys[m != 1] = -100
        return xs, ys


def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    return xs, ys


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def get_lr(step, warmup, max_steps, base_lr, min_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step > max_steps:
        return min_lr
    p = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * p))


def main():
    print("Loading tokenizer ...")
    tok_path = os.path.join(MODEL_DIR, "tokenizer.json")
    if not os.path.exists(tok_path):
        print("Tokenizer not found - run train_tokenizer.py first.")
        sys.exit(1)
    tokenizer = Tokenizer.from_file(tok_path)

    cfg = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_size=256,
        num_layers=8,
        num_heads=4,
        ffn_size=512,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
        pad_id=tokenizer.token_to_id("<pad>"),
        bos_id=tokenizer.token_to_id("<bos>"),
        eos_id=tokenizer.token_to_id("<eos>"),
        unk_id=tokenizer.token_to_id("<unk>"),
    )

    print("Loading dataset ...")
    all_examples = load_examples(tokenizer, DATA_PATH)
    print(f"Loaded {len(all_examples)} chat examples.")

    random.shuffle(all_examples)
    n_val = max(1, int(len(all_examples) * VAL_RATIO))
    train_examples = all_examples[n_val:]
    val_examples = all_examples[:n_val]

    train_ds = ChatDataset(tokenizer, train_examples, max_len=MAX_SEQ_LEN)
    val_ds = ChatDataset(tokenizer, val_examples, max_len=MAX_SEQ_LEN)
    print(f"Train sequences: {len(train_ds)}, val sequences: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    model = BitNetTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)

    cfg.to_json(os.path.join(MODEL_DIR, "config.json"))
    print(f"Config saved to {os.path.join(MODEL_DIR, 'config.json')}")

    step = 0
    it = iter(train_loader)
    best_val = float("inf")
    start_time = time.time()

    model.train()
    while step < MAX_STEPS:
        try:
            xs, ys = next(it)
        except StopIteration:
            it = iter(train_loader)
            xs, ys = next(it)

        lr = get_lr(step, WARMUP_STEPS, MAX_STEPS, LEARNING_RATE, MIN_LR)
        for g in optimizer.param_groups:
            g["lr"] = lr

        xs, ys = xs.to(device), ys.to(device)

        # Skip batches with no trainable tokens (all -100) to avoid NaN loss.
        if (ys != -100).sum().item() == 0:
            continue

        logits, _ = model(xs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), ys.view(-1), ignore_index=-100) / GRAD_ACCUM

        loss.backward()

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % LOG_EVERY == 0:
            el = time.time() - start_time
            print(f"step {step:5d} | lr {lr:.2e} | loss {loss.item() * GRAD_ACCUM:.4f} | {el:.1f}s")

        if step % EVAL_EVERY == 0 and step > 0:
            model.eval()
            total = 0
            n_tok = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    _, vl = model(vx, vy)
                    total += vl.item() * vx.size(0)
                    n_tok += vx.size(0)
            val_loss = total / max(1, n_tok)
            print(f"  == eval @ step {step}: val loss {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                ckpt = os.path.join(MODEL_DIR, "best.pt")
                torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, ckpt)
                print(f"  => saved best checkpoint {ckpt}")
            model.train()

        if step % SAVE_EVERY == 0 and step > 0:
            ckpt = os.path.join(MODEL_DIR, f"checkpoint_{step:05d}.pt")
            torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, ckpt)
            print(f"  => saved {ckpt}")

        step += 1

    # final save
    ckpt = os.path.join(MODEL_DIR, "final.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, ckpt)
    print(f"Done. Final checkpoint saved to {ckpt}")


if __name__ == "__main__":
    main()