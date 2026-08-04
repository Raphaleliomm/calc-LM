"""Evaluate the trained Recurrent Transformer (AttnRes) on the held-out no-world-knowledge split."""

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from model import ModelConfig, RecurrentTransformer

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
SEED = 42


def load_sequences(tokenizer, path, max_len=64, num_examples=200, seed=SEED):
    """Load chat examples, build shifted causal-LM sequences (same as train.py)."""
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")

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
                    ids.extend([bos_id] + enc.ids)
                    mask.extend([-100] + [-100] * len(enc.ids))
                elif role == "assistant":
                    ids.extend(enc.ids + [eos_id])
                    mask.extend([1] * len(enc.ids) + [1])
                else:
                    ids.extend(enc.ids)
                    mask.extend([-100] * len(enc.ids))

            if ids:
                examples.append((ids, mask))

    random.seed(seed)
    random.shuffle(examples)
    examples = examples[:num_examples]

    seqs = []
    for ids, mask in examples:
        for start in range(0, len(ids), max_len):
            in_ids = ids[start:start + max_len]
            tgt_ids = ids[start + 1:start + max_len + 1]
            tgt_mask = mask[start + 1:start + max_len + 1]
            if not tgt_ids:
                continue
            if len(in_ids) < max_len:
                in_ids = in_ids + [pad_id] * (max_len - len(in_ids))
            if len(tgt_ids) < max_len:
                tgt_ids = tgt_ids + [-100] * (max_len - len(tgt_ids))
                tgt_mask = tgt_mask + [-100] * (max_len - len(tgt_mask))
            xs = torch.tensor(in_ids, dtype=torch.long)
            ys = torch.tensor(tgt_ids, dtype=torch.long)
            m = torch.tensor(tgt_mask, dtype=torch.long)
            ys[m != 1] = -100
            seqs.append((xs, ys))
    return seqs


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_pred = 0
    for xs, ys in loader:
        xs, ys = xs.to(device), ys.to(device)
        logits, _ = model(xs)
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), ys.view(-1), ignore_index=-100)
        n_tokens = (ys != -100).sum().item()
        total_loss += loss.item() * max(1, n_tokens)
        total_tokens += n_tokens

        pred = logits.view(-1, logits.size(-1)).argmax(dim=-1)
        tgt = ys.view(-1)
        valid = tgt != -100
        total_correct += (pred[valid] == tgt[valid]).sum().item()
        total_pred += valid.sum().item()

    avg_loss = total_loss / max(1, total_tokens)
    acc = total_correct / max(1, total_pred)
    ppl = float(torch.exp(torch.tensor(avg_loss)).item())
    return avg_loss, ppl, acc


def main():
    parser = argparse.ArgumentParser(description="Eval Recurrent Transformer (AttnRes)")
    parser.add_argument("--ckpt", default="best.pt", help="Checkpoint file name (default: best.pt)")
    parser.add_argument("--num-examples", type=int, default=200,
                        help="Number of holdout chat examples to evaluate (default: 200)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    cfg = ModelConfig.from_json(os.path.join(MODEL_DIR, "config.json"))

    model = RecurrentTransformer(cfg)
    state = torch.load(os.path.join(MODEL_DIR, args.ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Loaded {args.ckpt} on {device}")

    print(f"Loading dataset, {args.num_examples} holdout examples ...")
    seqs = load_sequences(tokenizer, DATA_PATH, max_len=MAX_SEQ_LEN,
                          num_examples=args.num_examples, seed=args.seed)
    loader = DataLoader(seqs, batch_size=BATCH_SIZE, shuffle=False)
    print(f"{len(seqs)} sequences.")

    loss, ppl, acc = evaluate(model, loader, device)
    print("-" * 50)
    print(f"Val loss: {loss:.4f}")
    print(f"Perplexity: {ppl:.4f}")
    print(f"Token accuracy: {acc * 100:.2f}%")
    print(f"Checkpoint: {args.ckpt}")


if __name__ == "__main__":
    main()