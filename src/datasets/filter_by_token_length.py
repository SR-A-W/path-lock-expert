#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter over-long data with the real tokenizer

Candidates with chars > SAFE_CHARS_THRESHOLD are tokenized one by one,
and samples exceeding max_tokens are dropped. Samples below the threshold are kept as-is (safe).

Tokenization is parallelized across worker processes.

Usage:

    # Filter a single file
    python src/datasets/filter_by_token_length.py \
        --input train/data/superior_reasoning_27k_27k.jsonl \
        --output train/data/superior_reasoning_27k_27k_filtered.jsonl \
        --max_tokens 43008 \
        --safe_chars 50000 \
        --workers 16

    # Filter several files
    python src/datasets/filter_by_token_length.py \
        --input train/data/superior_reasoning_27k_27k.jsonl train/data/superior_hybrid_37k_37k.jsonl \
        --max_tokens 43008 \
        --safe_chars 50000 \
        --workers 16
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Global tokenizer (lazily initialized in each worker process)
_tokenizer = None
_tokenizer_path = None


def init_tokenizer(model_path):
    """Initialize the tokenizer in each worker process"""
    global _tokenizer, _tokenizer_path
    if _tokenizer is None or _tokenizer_path != model_path:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _tokenizer_path = model_path


def count_tokens(text, model_path):
    """Count tokens of one text (runs in a worker process)"""
    init_tokenizer(model_path)
    return len(_tokenizer.encode(text, add_special_tokens=False))


def process_candidate(args):
    """Process one candidate item, return (index, token_count)"""
    idx, text, model_path = args
    tok = count_tokens(text, model_path)
    return (idx, tok)


def main():
    parser = argparse.ArgumentParser(description="Filter over-long data with the real tokenizer")
    parser.add_argument(
        "--input", type=str, nargs="+", required=True,
        help="Input JSONL file path(s) (multiple allowed)",
    )
    parser.add_argument(
        "--output_suffix", type=str, default="_filtered",
        help="Output file name suffix (default: _filtered, i.e. xxx.jsonl -> xxx_filtered.jsonl)",
    )
    parser.add_argument(
        "--inplace", action="store_true",
        help="Overwrite the input file in place (no suffix)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=43008,
        help="Maximum token threshold (default: 43008)",
    )
    parser.add_argument(
        "--safe_chars", type=int, default=50000,
        help="Safe character threshold; items below it are kept without tokenizing (default: 50000)",
    )
    parser.add_argument(
        "--model_path", type=str,
        default="./models/source/Qwen/Qwen3-4B-Instruct-2507",
        help="Tokenizer model path",
    )
    parser.add_argument(
        "--workers", type=int, default=16,
        help="Number of parallel workers (default: 16)",
    )
    args = parser.parse_args()

    for input_path in args.input:
        input_path = Path(input_path)
        if not input_path.exists():
            print(f"ERROR: {input_path} does not exist")
            continue

        if args.inplace:
            output_path = input_path
        else:
            output_path = input_path.parent / (input_path.stem + args.output_suffix + ".jsonl")

        print(f"\n{'='*60}")
        print(f"Filtering: {input_path}")
        print(f"Output: {output_path}")
        print(f"max_tokens: {args.max_tokens}, safe_chars: {args.safe_chars}, workers: {args.workers}")
        print(f"{'='*60}")

        # 1. Load data, split into safe items and candidates
        all_items = []
        safe_indices = []
        candidate_tasks = []

        start_time = time.time()
        with open(input_path) as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                item = json.loads(line)
                all_items.append(item)
                u = next(c["value"] for c in item["conversations"] if c["from"] == "user")
                a = next(c["value"] for c in item["conversations"] if c["from"] == "assistant")
                chars = len(u) + len(a)

                if chars <= args.safe_chars:
                    safe_indices.append(idx)
                else:
                    candidate_tasks.append((idx, u + a, args.model_path))

        total = len(all_items)
        print(f"  Total: {total}")
        print(f"  Safe (chars <= {args.safe_chars}): {len(safe_indices)}")
        print(f"  Candidates (to tokenize): {len(candidate_tasks)}")

        # 2. Tokenize candidates in parallel
        print(f"  Tokenizing {len(candidate_tasks)} candidates with {args.workers} workers...")
        keep_set = set(safe_indices)
        removed = 0
        removed_details = []

        if candidate_tasks:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process_candidate, task): task[0] for task in candidate_tasks}
                done = 0
                for future in as_completed(futures):
                    idx, tok = future.result()
                    done += 1
                    if tok <= args.max_tokens:
                        keep_set.add(idx)
                    else:
                        removed += 1
                        removed_details.append((idx, tok))
                    if done % 500 == 0 or done == len(candidate_tasks):
                        elapsed = time.time() - start_time
                        print(f"    [{done}/{len(candidate_tasks)}] removed so far: {removed}, elapsed: {elapsed:.1f}s")

        # 3. Write results
        kept_items = [all_items[i] for i in sorted(keep_set)]

        # If inplace, write a temporary file first and then replace
        if args.inplace:
            tmp_path = str(output_path) + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                for item in kept_items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            os.replace(tmp_path, str(output_path))
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                for item in kept_items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        elapsed = time.time() - start_time
        print(f"\n  Result: {total} -> {len(kept_items)} (removed {removed})")
        print(f"  Elapsed: {elapsed:.1f}s")

        if removed_details:
            removed_details.sort(key=lambda x: -x[1])
            print(f"  Longest 5 removed items:")
            for idx, tok in removed_details[:5]:
                print(f"    index={idx}, tokens={tok}")


if __name__ == "__main__":
    main()
