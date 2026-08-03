"""Count total tokens in the 'no world knowledge' classification output."""

import json
import sys

JSONL_PATH = r"d:\Github\calc-LM\kaggle_output\world_knowledge_classification\rank0_no_world_knowledge_confidence_5_plus.jsonl"


def extract_text(sample_json_str):
    """Extract all human-readable text from a dataset sample.

    The sample is a JSON string like:
      {"messages": [{"content": "...", "role": "user"}, {"content": "...", "role": "assistant"}]}
    We extract all 'content' values and concatenate them.
    """
    try:
        data = json.loads(sample_json_str)
    except (json.JSONDecodeError, TypeError):
        # If it's not valid JSON, just return the raw string
        return sample_json_str

    texts = []
    if isinstance(data, dict):
        # Check for "messages" format (chat data)
        if "messages" in data:
            for msg in data["messages"]:
                if isinstance(msg, dict) and "content" in msg:
                    texts.append(str(msg["content"]))
        # Check for other common fields
        for key in ("text", "question", "answer", "prompt", "response", "instruction", "output", "input"):
            if key in data:
                texts.append(str(data[key]))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, str):
                        texts.append(v)
    else:
        texts.append(str(data))

    return "\n".join(texts)


def main():
    # Try to use a real tokenizer
    tokenizer = None
    tokenizer_name = "unknown"
    try:
        from transformers import AutoTokenizer
        # Try GPT-2 as a BPE approximation (similar vocab size to Llama/Mistral)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer_name = "GPT-2 BPE"
    except Exception:
        try:
            from transformers import GPT2TokenizerFast
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            tokenizer_name = "GPT-2 BPE (fast)"
        except Exception:
            pass

    total_chars = 0
    total_tokens = 0
    total_rows = 0
    total_sample_chars = 0  # chars in the raw "sample" field (including JSON overhead)

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_rows += 1
            sample_raw = record.get("sample", "")
            total_sample_chars += len(sample_raw)

            # Extract just the text content (no JSON overhead)
            text = extract_text(sample_raw)
            total_chars += len(text)

            if tokenizer is not None:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                total_tokens += len(tokens)

    print(f"=== Token Count Report ===")
    print(f"File: {JSONL_PATH}")
    print(f"Rows: {total_rows:,}")
    print(f"")
    print(f"Text content (extracted from messages):")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Avg chars/row:    {total_chars / max(total_rows, 1):,.0f}")
    print(f"")

    if tokenizer is not None:
        print(f"Tokenizer: {tokenizer_name}")
        print(f"  Total tokens:     {total_tokens:,}")
        print(f"  Avg tokens/row:   {total_tokens / max(total_rows, 1):,.0f}")
        print(f"  Tokens per char:  {total_tokens / max(total_chars, 1):.3f}")
    else:
        # Rough estimate: ~4 chars per token for English/German text
        estimate = total_chars // 4
        print(f"Tokenizer: none available (using estimate ~4 chars/token)")
        print(f"  Estimated tokens: {estimate:,}")
        total_tokens = estimate

    print(f"")
    print(f"Raw 'sample' field (including JSON overhead):")
    print(f"  Total characters: {total_sample_chars:,}")
    raw_estimate = total_sample_chars // 4
    print(f"  Estimated tokens: {raw_estimate:,} (incl. JSON overhead)")
    print(f"")
    print(f"=== Summary ===")
    print(f"Training tokens (text only):  ~{total_tokens:,}")
    print(f"Training tokens (with JSON):   ~{raw_estimate:,}")
    print(f"Rows:                          {total_rows:,}")


if __name__ == "__main__":
    main()