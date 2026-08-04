"""Run inference with the trained Recurrent Transformer (AttnRes)."""

import argparse
import os
import sys

import torch
from tokenizers import Tokenizer

# Ensure UTF-8 output for ByteLevel tokenizer Unicode chars on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from model import ModelConfig, RecurrentTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "checkpoints"))


def load_model(ckpt_name="best.pt"):
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    cfg = ModelConfig.from_json(os.path.join(MODEL_DIR, "config.json"))

    model = RecurrentTransformer(cfg)
    ckpt_path = os.path.join(MODEL_DIR, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tok, device


def main():
    parser = argparse.ArgumentParser(description="Inference for Recurrent Transformer (AttnRes)")
    parser.add_argument("prompt", nargs="?", default=None,
                        help="Optional prompt. If omitted, interactive mode is used.")
    parser.add_argument("--ckpt", default="best.pt", help="Checkpoint file name (default: best.pt)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    model, tok, device = load_model(args.ckpt)
    print(f"Loaded {args.ckpt} on {device}")

    def run(prompt):
        print("\n" + "=" * 60)
        print("PROMPT:")
        print(prompt)
        print("-" * 60)
        out = model.generate(
            tok,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print("ANSWER:")
        print(out)
        print("=" * 60)

    if args.prompt:
        run(args.prompt)
    else:
        print("Interactive mode - type 'exit' or Ctrl+C to quit.")
        while True:
            try:
                prompt = input("\n>>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit"):
                break
            run(prompt)


if __name__ == "__main__":
    main()