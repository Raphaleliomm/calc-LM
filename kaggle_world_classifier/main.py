"""Classify every row of the requested Hugging Face datasets on one GPU.

The script intentionally performs one `model.generate` call for every input
record.  It never batches records together, so every output row is traceable to
one fresh model call.
"""

import gc
import json
import multiprocessing as mp
import os
import queue
import re
import subprocess
import sys
import time
import argparse
from pathlib import Path


# Keep Hugging Face caches in the output volume rather than the small root disk.
os.environ.setdefault("HF_HOME", "/kaggle/working/hf-cache")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Avoid allocator fragmentation on long-running Kaggle sessions. Set both
# spellings because the supported name depends on the installed PyTorch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
OUTPUT_DIR = Path(os.environ.get("WORLD_KNOWLEDGE_OUTPUT_DIR", "/kaggle/working/world_knowledge_classification"))
NUM_GPUS = 1
MAX_NEW_TOKENS = 1
MAX_INPUT_TOKENS = 2048

DATASETS = [
    "ianncity/GLM-5.2-Conversation",
    "sanjay-29-29/math-dataset-instruction",
    "MU-NLPC/Calc-gsm8k",
    "D3xter1922/proofwriter-dataset",
    "meta-math/MetaMathQA",
]

SYSTEM_PROMPT = r'''Classify if the text contains WORLD KNOWLEDGE (real-world facts: history, geography, science, real people/organizations/brands).

Answer with ONLY one word: "unsafe" (contains world knowledge) or "safe" (pure math, logic, fiction, or conversation).

Rules:
- Math with fictional characters (Alice has 3 apples) = safe
- Math mentioning real places/people (Paris, Einstein) = unsafe
- Logic puzzles with abstract names (Alice, Bob) = safe
- If unsure = unsafe
'''

CATEGORY_FILENAMES = {
    ("unsafe", "5_plus"): "world_knowledge_confidence_5_plus.jsonl",
    ("unsafe", "4_minus"): "world_knowledge_confidence_4_minus.jsonl",
    ("safe", "5_plus"): "no_world_knowledge_confidence_5_plus.jsonl",
    ("safe", "4_minus"): "no_world_knowledge_confidence_4_minus.jsonl",
}


def build_forced_choice_processor(tokenizer):
    """Build a LogitsProcessor that forces the model to output only 'safe' or 'unsafe'.

    The processor masks every token except the first token of 'safe' or 'unsafe'
    (with and without leading space, capitalised variants).  Combined with
    ``max_new_tokens=1`` this gives a true single-token, no-reasoning answer.
    """
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    token_to_label = {}
    for label in ("safe", "unsafe"):
        for prefix in ("", " "):
            for word_form in (label, label.capitalize()):
                token_ids = tokenizer.encode(prefix + word_form, add_special_tokens=False)
                if token_ids:
                    token_to_label[token_ids[0]] = label

    allowed_token_ids = list(token_to_label.keys())

    class _ForcedChoiceProcessor(LogitsProcessor):
        def __call__(self, input_ids, scores):
            mask = torch.full_like(scores, float("-inf"))
            for token_id in allowed_token_ids:
                mask[:, token_id] = 0
            return scores + mask

    return LogitsProcessorList([_ForcedChoiceProcessor()]), token_to_label


def example_key(record):
    """Stable identity used to resume without writing duplicate examples."""
    return "\x1f".join(
        [
            str(record["source_dataset"]),
            str(record["config"]),
            str(record["split"]),
            str(record["row_index"]),
        ]
    )


def load_completed_keys():
    """Read rows that completed inference successfully in earlier runs.

    Failed rows are deliberately excluded so a rerun after a transient model or
    prompt problem retries them instead of treating the conservative fallback as
    a permanent result.
    """
    completed = set()
    if not OUTPUT_DIR.exists():
        return completed
    for path in OUTPUT_DIR.glob("*.jsonl"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                        if (
                            all(field in record for field in ("source_dataset", "config", "split", "row_index"))
                            and not str(record.get("parse_status", "")).startswith("inference_error:")
                        ):
                            completed.add(example_key(record))
                    except (json.JSONDecodeError, TypeError, KeyError):
                        # A process can be killed while writing its final line.
                        continue
        except OSError as exc:
            print(f"CHECKPOINT_READ_WARNING path={path} error={exc}", flush=True)
    return completed


def save_progress(progress):
    """Atomically persist a small progress checkpoint for the Kaggle output."""
    target = OUTPUT_DIR / "progress.json"
    temporary = OUTPUT_DIR / "progress.json.tmp"
    temporary.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    temporary.replace(target)


def install_dependencies():
    """Pin versions new enough to support Mistral 7B and 4-bit loading."""
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--upgrade",
            "transformers>=4.48.0",
            "accelerate>=1.3.0",
            "bitsandbytes>=0.45.2",
            "datasets>=3.2.0",
            "huggingface_hub>=0.27.0",
        ]
    )


def normalize_prediction(decoded, token_id=None, token_to_label=None):
    """Map a single-token answer to a classification, defaulting conservatively."""
    if token_to_label and token_id is not None and token_id in token_to_label:
        return {
            "classification": token_to_label[token_id],
            "reason": "Forced single-token classification.",
            "confidence": 5,
            "parse_status": "ok",
            "raw_model_output": decoded,
        }
    # Fallback: parse the decoded text
    decoded_lower = decoded.strip().lower()
    if "unsafe" in decoded_lower:
        classification = "unsafe"
    elif "safe" in decoded_lower:
        classification = "safe"
    else:
        classification = "unsafe"
    return {
        "classification": classification,
        "reason": "Forced single-token classification (fallback parsing).",
        "confidence": 5,
        "parse_status": "ok" if decoded_lower else "empty_output",
        "raw_model_output": decoded,
    }


def category_for(prediction):
    confidence_band = "5_plus" if prediction["confidence"] >= 5 else "4_minus"
    return CATEGORY_FILENAMES[(prediction["classification"], confidence_band)]


def load_model(gpu_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.cuda.set_device(gpu_id)
    # The Kaggle model source is mounted read-only under /kaggle/input. The
    # fp16 checkpoint is ~94 GB unpacked, so never download it to /kaggle/working.
    model_path = None
    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates = sorted(
            p.parent
            for p in input_root.rglob("config.json")
            if "mistral" in str(p).lower() and (p.parent / "tokenizer.json").exists()
        )
        if candidates:
            model_path = str(candidates[0])
    if model_path is None:
        raise RuntimeError(
            "The Kaggle Mistral 7B model input was not mounted. Add model source "
            "mistral-ai/mistral/PyTorch/7b-instruct-v0.1-hf/1."
        )
    print(f"Loading local Kaggle model from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, fix_mistral_regex=True, local_files_only=True
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map={"": gpu_id},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer


def classify_one(model, tokenizer, gpu_id, sample_text):
    """Exactly one fresh generate call is made for this dataset example.

    The model is forced to output a single token ('safe' or 'unsafe') via a
    LogitsProcessor that masks all other tokens.  This eliminates reasoning
    / chain-of-thought output and makes each call near-instant.
    """
    import torch

    processor, token_to_label = build_forced_choice_processor(tokenizer)

    # Mistral-7B-Instruct-v0.1 only accepts alternating user/assistant turns;
    # unlike newer chat models, its bundled template rejects a separate system
    # role. Put the instructions in the initial user turn so the prompt is
    # valid for this specific checkpoint.
    messages = [
        {
            "role": "user",
            "content": (
                SYSTEM_PROMPT
                + "\n\nClassify this dataset sample (answer ONLY 'safe' or 'unsafe'):\n"
                + sample_text
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # A malformed/very long row must never be allowed to create an unbounded
    # attention matrix. Retry only the current row with a shorter prompt after
    # an OOM; rows are never batched together.
    attempts = ((MAX_INPUT_TOKENS, MAX_NEW_TOKENS), (1024, MAX_NEW_TOKENS), (512, MAX_NEW_TOKENS))
    for input_limit, output_limit in attempts:
        encoded = None
        generated = None
        try:
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=input_limit,
            )
            encoded = {name: tensor.to(f"cuda:{gpu_id}") for name, tensor in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=output_limit,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    logits_processor=processor,
                )
            answer_tokens = generated[0, encoded["input_ids"].shape[1] :]
            decoded = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            first_token_id = answer_tokens[0].item() if len(answer_tokens) > 0 else None
            prediction = normalize_prediction(decoded, first_token_id, token_to_label)
            if input_limit != MAX_INPUT_TOKENS:
                prediction["parse_status"] = f"{prediction['parse_status']}; oom_retry_input_tokens={input_limit}"
            return prediction
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if input_limit == attempts[-1][0]:
                return {
                    "classification": "unsafe",
                    "reason": "CUDA memory remained insufficient after bounded retries; conservative unsafe fallback applied.",
                    "confidence": 5,
                    "parse_status": "oom_fallback",
                    "raw_model_output": "",
                }
            print(
                f"OOM_RETRY gpu={gpu_id} input_tokens={input_limit} next_input_tokens={attempts[attempts.index((input_limit, output_limit)) + 1][0]}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            del encoded, generated
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise RuntimeError("unreachable")


def worker(gpu_id, jobs, completed):
    """Own one GPU and one Mistral instance; write an independent rank shard."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"GPU_WORKER_START gpu={gpu_id}", flush=True)
    model, tokenizer = load_model(gpu_id)
    print(f"GPU_WORKER_READY gpu={gpu_id}", flush=True)
    completed.put({"kind": "ready", "gpu": gpu_id})
    handles = {
        filename: (OUTPUT_DIR / f"rank{gpu_id}_{filename}").open("a", encoding="utf-8")
        for filename in CATEGORY_FILENAMES.values()
    }
    try:
        while True:
            job = jobs.get()
            if job is None:
                break
            try:
                prediction = classify_one(model, tokenizer, gpu_id, job["sample"])
                record = {
                    "source_dataset": job["source_dataset"],
                    "config": job["config"],
                    "split": job["split"],
                    "row_index": job["row_index"],
                    "sample": job["sample"],
                    "classification": prediction["classification"],
                    "reason": prediction["reason"],
                    "confidence": prediction["confidence"],
                    "parse_status": prediction["parse_status"],
                    "raw_model_output": prediction["raw_model_output"],
                    "model": MODEL_ID,
                    "example_id": example_key(job),
                }
                handle = handles[category_for(prediction)]
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()  # retain completed work if the session is interrupted
                completed.put({"kind": "ok", "gpu": gpu_id})
            except Exception as exc:  # a failed row is still emitted conservatively
                record = {
                    **job,
                    "classification": "unsafe",
                    "reason": "Inference failed, so the conservative unsafe rule was applied.",
                    "confidence": 5,
                    "parse_status": f"inference_error: {type(exc).__name__}: {exc}",
                    "raw_model_output": "",
                    "model": MODEL_ID,
                    "example_id": example_key(job),
                }
                handles[CATEGORY_FILENAMES[("unsafe", "5_plus")]].write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                handles[CATEGORY_FILENAMES[("unsafe", "5_plus")]].flush()
                completed.put({"kind": "error", "gpu": gpu_id, "error": str(exc)})
            finally:
                gc.collect()
    finally:
        for handle in handles.values():
            handle.close()


def iter_all_rows(dataset_ids):
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

    for dataset_id in dataset_ids:
        configs = get_dataset_config_names(dataset_id)
        # A repository with no named builder config is represented as "default".
        if not configs:
            configs = [None]
        for config in configs:
            splits = get_dataset_split_names(dataset_id, config_name=config)
            for split in splits:
                print(f"Streaming {dataset_id} | {config or 'default'} | {split}", flush=True)
                stream = load_dataset(dataset_id, name=config, split=split, streaming=True)
                for row_index, row in enumerate(stream):
                    yield {
                        "source_dataset": dataset_id,
                        "config": config or "default",
                        "split": split,
                        "row_index": row_index,
                        "sample": json.dumps(row, ensure_ascii=False, sort_keys=True, default=str),
                    }


def merge_rank_shards():
    """Produce deduplicated final files, preferring a later successful retry."""
    records_by_key = {}
    for filename in CATEGORY_FILENAMES.values():
        for gpu_id in range(NUM_GPUS):
            shard = OUTPUT_DIR / f"rank{gpu_id}_{filename}"
            if not shard.exists():
                continue
            with shard.open("r", encoding="utf-8") as source:
                for line in source:
                    try:
                        record = json.loads(line)
                        key = record["example_id"]
                    except (json.JSONDecodeError, KeyError, TypeError):
                        # Preserve any partial/corrupt line only in its rank
                        # shard; final outputs contain valid JSONL.
                        continue
                    existing = records_by_key.get(key)
                    if existing is None or (
                        str(existing.get("parse_status", "")).startswith("inference_error:")
                        and not str(record.get("parse_status", "")).startswith("inference_error:")
                    ):
                        records_by_key[key] = record

    final_records = {filename: [] for filename in CATEGORY_FILENAMES.values()}
    for record in records_by_key.values():
        final_records[category_for(record)].append(record)
    for filename, records in final_records.items():
        with (OUTPUT_DIR / filename).open("w", encoding="utf-8") as destination:
            for record in records:
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Classify dataset rows one by one.")
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        choices=DATASETS,
        help="Dataset to process; repeat for multiple datasets (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Checkpoint/output directory (default: Kaggle working directory).",
    )
    # parse_known_args ignores unrecognised flags that Jupyter/Kaggle/Colab
    # kernels inject into sys.argv (e.g. -f /tmp/...json --HistoryManager...).
    return parser.parse_known_args()[0]


def main():
    global OUTPUT_DIR
    args = parse_args()
    OUTPUT_DIR = args.output_dir
    dataset_ids = args.datasets or DATASETS
    install_dependencies()
    import torch

    if torch.cuda.device_count() < NUM_GPUS:
        raise RuntimeError(
            f"This run requires at least {NUM_GPUS} visible GPU; found {torch.cuda.device_count()}. "
            "Select a Kaggle GPU, then rerun."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Restore checkpoint from the Kaggle input dataset so the script resumes
    # where the previous run left off instead of starting from scratch.
    checkpoint_input = Path("/kaggle/input/world-knowledge-checkpoint")
    if checkpoint_input.exists():
        import shutil
        for src_file in checkpoint_input.glob("*.jsonl"):
            dest_file = OUTPUT_DIR / src_file.name
            if not dest_file.exists():
                print(f"RESTORE_CHECKPOINT {src_file.name}", flush=True)
                shutil.copy2(src_file, dest_file)
        print("Checkpoint restored from input dataset.", flush=True)

    started_at = time.time()
    completed_keys = load_completed_keys()
    print(
        f"RESUME_CHECK completed_examples={len(completed_keys):,} "
        f"checkpoint_dir={OUTPUT_DIR}",
        flush=True,
    )
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(
            {
                "model": MODEL_ID,
                "datasets": dataset_ids,
                "num_gpus": NUM_GPUS,
                "one_model_call_per_example": True,
                "forced_single_token": True,
                "max_new_tokens": MAX_NEW_TOKENS,
                "started_at_unix": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    ctx = mp.get_context("spawn")
    jobs = ctx.Queue(maxsize=32)
    completed = ctx.Queue()
    processes = [ctx.Process(target=worker, args=(gpu_id, jobs, completed)) for gpu_id in range(NUM_GPUS)]
    for process in processes:
        process.start()
    print(f"WORKERS_STARTED count={len(processes)}; waiting for model readiness", flush=True)

    seen = skipped = submitted = successful = errors = ready = 0
    last_reported_completed = 0

    def drain_results():
        nonlocal successful, errors, ready
        while True:
            try:
                result = completed.get_nowait()
            except queue.Empty:
                return
            kind = result.get("kind")
            if kind == "ok":
                successful += 1
            elif kind == "error":
                errors += 1
                print(
                    f"INFERENCE_ERROR gpu={result.get('gpu')} error={result.get('error')}",
                    flush=True,
                )
            elif kind == "ready":
                ready += 1
                print(f"MODEL_READY gpu={result.get('gpu')}", flush=True)

    def report_completed(force=False):
        nonlocal last_reported_completed
        drain_results()
        classified = successful + errors
        if not force and classified < last_reported_completed + 100:
            return
        if classified == last_reported_completed and not force:
            return
        last_reported_completed = classified
        progress = {
            "elapsed_seconds": int(time.time() - started_at),
            "ready_workers": ready,
            "seen_rows": seen,
            "skipped_existing": skipped,
            "submitted_for_inference": submitted,
            "completed_successfully": successful,
            "inference_errors": errors,
            "updated_at_unix": time.time(),
        }
        print(
            "PROGRESS "
            + " ".join(f"{key}={value}" for key, value in progress.items()),
            flush=True,
        )
        save_progress(progress)

    def assert_workers_alive():
        dead = [process.exitcode for process in processes if not process.is_alive() and process.exitcode is not None]
        if dead:
            raise RuntimeError(f"Inference worker exited unexpectedly with codes={dead}")

    try:
        for job in iter_all_rows(dataset_ids):
            seen += 1
            key = example_key(job)
            if key in completed_keys:
                skipped += 1
                continue
            while True:
                try:
                    jobs.put(job, timeout=5)
                    break
                except queue.Full:
                    drain_results()
                    assert_workers_alive()
                    report_completed()
            submitted += 1
            drain_results()
            report_completed()
    finally:
        for _ in processes:
            while True:
                try:
                    jobs.put(None, timeout=5)
                    break
                except queue.Full:
                    drain_results()
                    if not any(process.is_alive() for process in processes):
                        break
        for process in processes:
            process.join()

    drain_results()
    report_completed(force=True)
    merge_rank_shards()
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(
            {
                "seen": seen,
                "skipped_existing": skipped,
                "submitted": submitted,
                "successful": successful,
                "errors": errors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Finished. Outputs: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
