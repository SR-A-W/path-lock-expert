#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge /think and /no_think data into the final training set

Reads think_data.jsonl (from prepare_think_data.py) and
no_think_filtered.jsonl (from filter_no_think_data.py),
samples them at the given ratio and writes the final training JSONL.

No shuffling is done (LLaMA-Factory's RandomSampler handles it).

Several ratio variants (e.g. 1:1, 1:2, 2:1) can be generated in one run.

Output goes to datasets/superior_reasoning/final/; copy it to train/data/ and register it manually.

Usage:
    python -m src.datasets.merge_hybrid_dataset \
        --think_input datasets/superior_reasoning/think_data.jsonl \
        --no_think_input datasets/superior_reasoning/no_think_filtered.jsonl \
        --output_dir datasets/superior_reasoning/final \
        --ratios 1:1,1:2,2:1

Dependencies:
    None beyond the standard library
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Project root directory
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ==================== Helper functions ====================

def load_jsonl(path: str) -> list:
    """Load a JSONL file"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(data: list, path: str):
    """Write a JSONL file"""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_ratio(ratio_str: str) -> tuple[int, int]:
    """Parse a ratio string (e.g. '1:2' -> (1, 2))"""
    parts = ratio_str.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid ratio format: {ratio_str}; expected 'X:Y'")
    return int(parts[0]), int(parts[1])


def compute_counts(
    n_think: int, n_no_think: int, ratio_think: int, ratio_no_think: int
) -> tuple[int, int]:
    """Compute the actual sample counts from the ratio and the available data

    The side with less data is the bottleneck; the other side is scaled by the ratio.

    Args:
        n_think: number of available think samples
        n_no_think: number of available no_think samples
        ratio_think: think ratio coefficient
        ratio_no_think: no_think ratio coefficient

    Returns:
        (think_count, no_think_count) actual sample counts
    """
    per_unit_think = n_think / ratio_think
    per_unit_no_think = n_no_think / ratio_no_think
    per_unit = min(per_unit_think, per_unit_no_think)

    think_count = int(per_unit * ratio_think)
    no_think_count = int(per_unit * ratio_no_think)

    return think_count, no_think_count


# dataset_info.json registration template
DATASET_INFO_TEMPLATE = {
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "system": None},
    "tags": {
        "role_tag": "from",
        "content_tag": "value",
        "user_tag": "user",
        "assistant_tag": "assistant",
    },
}


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Merge /think and /no_think data into the final training set"
    )
    parser.add_argument(
        "--think_input", type=str,
        default="datasets/superior_reasoning/think_data.jsonl",
        help="Path to the think data",
    )
    parser.add_argument(
        "--no_think_input", type=str,
        default="datasets/superior_reasoning/no_think_filtered.jsonl",
        help="Path to the no_think data",
    )
    parser.add_argument(
        "--output_dir", type=str, default="datasets/superior_reasoning/final",
        help="Output directory",
    )
    parser.add_argument(
        "--output_name", type=str, default="superior_reasoning",
        help="Output file name prefix",
    )
    parser.add_argument(
        "--ratios", type=str, default="1:1",
        help="think:no_think ratios, comma-separated (e.g. 1:1,1:2,2:1)",
    )
    parser.add_argument(
        "--max_total", type=int, default=0,
        help="Maximum total number of samples (0 = no limit)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (used for sampling)")
    args = parser.parse_args()

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ratios = [parse_ratio(r) for r in args.ratios.split(",")]

    print(f"\n{'='*60}")
    print("Step 4: Merge the final training set")
    print(f"{'='*60}")
    print(f"think data:    {args.think_input}")
    print(f"no_think data: {args.no_think_input}")
    print(f"Ratio variants: {args.ratios}")
    print(f"Random seed:    {args.seed}")
    print(f"Note: no shuffling (handled by LLaMA-Factory RandomSampler)")

    # ---- 1. Load data ----
    print(f"\n{'─'*40}")
    print("1. Loading data...")
    think_data = load_jsonl(str(project_root / args.think_input))
    no_think_data = load_jsonl(str(project_root / args.no_think_input))
    print(f"  think samples:    {len(think_data)}")
    print(f"  no_think samples: {len(no_think_data)}")

    # ---- 2. Build each ratio variant ----
    print(f"\n{'─'*40}")
    print("2. Generating ratio variants...")

    registry_entries = {}

    for rt, rn in ratios:
        ratio_label = f"{rt}_{rn}"
        tc, nc = compute_counts(len(think_data), len(no_think_data), rt, rn)

        # Optional: cap the total size
        if args.max_total > 0 and (tc + nc) > args.max_total:
            scale = args.max_total / (tc + nc)
            tc = int(tc * scale)
            nc = int(nc * scale)

        random.seed(args.seed)
        sampled_think = random.sample(think_data, tc)
        sampled_no_think = random.sample(no_think_data, nc)

        # No shuffle; concatenate directly (think first, then no_think)
        combined = sampled_think + sampled_no_think

        filename = f"{args.output_name}_{ratio_label}.jsonl"
        output_path = output_dir / filename
        write_jsonl(combined, str(output_path))

        print(f"\n  Ratio {rt}:{rn}:")
        print(f"    think: {tc}, no_think: {nc}, total: {tc + nc}")
        print(f"    output: {output_path}")

        # Registration entry
        entry_name = f"{args.output_name}_{ratio_label}"
        registry_entries[entry_name] = {
            "file_name": filename,
            **DATASET_INFO_TEMPLATE,
        }

    # ---- 3. Print registration entries ----
    print(f"\n{'─'*40}")
    print("3. dataset_info.json entries (add them manually to train/data/dataset_info.json):")
    print(json.dumps(registry_entries, indent=2, ensure_ascii=False))

    print(f"\n{'─'*40}")
    print("Note: copy the final files you need from")
    print(f"  {output_dir}/")
    print("to train/data/ and register them in dataset_info.json.")

    print(f"\n{'='*60}")
    print("Step 4 done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
