"""Build a Byte-Pair-Encoding tokenizer with 2048 vocab on the training data only."""

import json
import os

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = os.environ.get("TRAIN_DATA", "kaggle_output/world_knowledge_classification/rank0_no_world_knowledge_confidence_5_plus.jsonl")
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "checkpoints"))
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
VOCAB_SIZE = 2048

os.makedirs(MODEL_DIR, exist_ok=True)


def iter_texts():
    """Yield the raw text (user+assistant messages) from every record."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                sample = json.loads(record["sample"])
                for msg in sample.get("messages", []):
                    content = msg.get("content", "")
                    if content:
                        yield content
            except Exception:
                continue


def main():
    print(f"Building tokenizer from {DATA_PATH} ...")
    n = 0
    for _ in iter_texts():
        n += 1
    print(f"Collected {n} text spans.")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train_from_iterator(iter_texts(), trainer=trainer)

    # Keep the ByteLevel decoder so decode() converts byte-level chars back to text.
    tokenizer.decoder = ByteLevelDecoder(add_prefix_space=False)

    tok_path = os.path.join(MODEL_DIR, "tokenizer.json")
    tokenizer.save(tok_path)
    print(f"Tokenizer saved to {tok_path}")

    # Print token count sanity check
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    for st in SPECIAL_TOKENS:
        print(f"  {st}: id {tokenizer.token_to_id(st)}")


if __name__ == "__main__":
    main()