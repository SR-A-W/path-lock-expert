#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用真实 tokenizer 过滤超长数据

对 chars > SAFE_CHARS_THRESHOLD 的候选数据逐条 tokenize，
剔除超过 max_tokens 的样本。chars 低于阈值的直接保留（安全）。

使用多线程并行 tokenize 加速。

使用方法:
    conda activate gpt-oss

    # 过滤单个文件
    python src/datasets/filter_by_token_length.py \
        --input train/data/superior_reasoning_27k_27k.jsonl \
        --output train/data/superior_reasoning_27k_27k_filtered.jsonl \
        --max_tokens 43008 \
        --safe_chars 50000 \
        --workers 16

    # 批量过滤多个文件
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

# 全局 tokenizer（每个 worker 进程延迟初始化）
_tokenizer = None
_tokenizer_path = None


def init_tokenizer(model_path):
    """在每个 worker 进程中初始化 tokenizer"""
    global _tokenizer, _tokenizer_path
    if _tokenizer is None or _tokenizer_path != model_path:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _tokenizer_path = model_path


def count_tokens(text, model_path):
    """计算单条文本的 token 数（在 worker 进程中执行）"""
    init_tokenizer(model_path)
    return len(_tokenizer.encode(text, add_special_tokens=False))


def process_candidate(args):
    """处理一条候选数据，返回 (index, token_count)"""
    idx, text, model_path = args
    tok = count_tokens(text, model_path)
    return (idx, tok)


def main():
    parser = argparse.ArgumentParser(description="使用真实 tokenizer 过滤超长数据")
    parser.add_argument(
        "--input", type=str, nargs="+", required=True,
        help="输入 JSONL 文件路径（支持多个）",
    )
    parser.add_argument(
        "--output_suffix", type=str, default="_filtered",
        help="输出文件名后缀（默认: _filtered，即 xxx.jsonl → xxx_filtered.jsonl）",
    )
    parser.add_argument(
        "--inplace", action="store_true",
        help="原地覆盖输入文件（不加后缀）",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=43008,
        help="最大 token 数阈值（默认: 43008）",
    )
    parser.add_argument(
        "--safe_chars", type=int, default=50000,
        help="安全字符数阈值，低于此值的直接保留不 tokenize（默认: 50000）",
    )
    parser.add_argument(
        "--model_path", type=str,
        default="./models/source/Qwen/Qwen3-4B-Instruct-2507",
        help="Tokenizer 模型路径",
    )
    parser.add_argument(
        "--workers", type=int, default=16,
        help="并行 worker 数（默认: 16）",
    )
    args = parser.parse_args()

    for input_path in args.input:
        input_path = Path(input_path)
        if not input_path.exists():
            print(f"ERROR: {input_path} 不存在")
            continue

        if args.inplace:
            output_path = input_path
        else:
            output_path = input_path.parent / (input_path.stem + args.output_suffix + ".jsonl")

        print(f"\n{'='*60}")
        print(f"过滤: {input_path}")
        print(f"输出: {output_path}")
        print(f"max_tokens: {args.max_tokens}, safe_chars: {args.safe_chars}, workers: {args.workers}")
        print(f"{'='*60}")

        # 1. 加载数据，分为安全和候选
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
        print(f"  总计: {total}")
        print(f"  安全 (chars <= {args.safe_chars}): {len(safe_indices)}")
        print(f"  候选 (需 tokenize): {len(candidate_tasks)}")

        # 2. 多进程 tokenize 候选数据
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

        # 3. 写入结果
        kept_items = [all_items[i] for i in sorted(keep_set)]

        # 如果 inplace，先写临时文件再替换
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
        print(f"\n  结果: {total} → {len(kept_items)} (剔除 {removed})")
        print(f"  耗时: {elapsed:.1f}s")

        if removed_details:
            removed_details.sort(key=lambda x: -x[1])
            print(f"  被剔除的最长 5 条:")
            for idx, tok in removed_details[:5]:
                print(f"    index={idx}, tokens={tok}")


if __name__ == "__main__":
    main()
