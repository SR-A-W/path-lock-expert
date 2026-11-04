#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert the Superior-Reasoning-SFT dataset to the /think training format

Downloads the Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b dataset from HuggingFace,
filters the given domains (default: math + science + instruction-following), converts to the LLaMA-Factory
ShareGPT format, and extracts ground truth for the later correctness filtering of the /no_think data.

The instruction-following domain is special: the text after </think> is extracted directly as the /no_think answer.

Three output files (in datasets/superior_reasoning/):
  1. think_data.jsonl           — /think training data for all domains (ShareGPT format)
  2. metadata.jsonl             — math + science questions + ground truth (used by Step 2)
  3. no_think_instruction.jsonl — /no_think data extracted directly from the instruction domain (ShareGPT format)

Usage:
    python -m src.datasets.prepare_think_data \
        --output_dir datasets/superior_reasoning

Dependencies:
    pip install datasets
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Project root directory
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ==================== Ground-truth extraction ====================

def extract_boxed_answer(text: str) -> str | None:
    """Extract the answer inside \\boxed{...} from text (nested braces supported)

    Uses a stack-matching algorithm for nesting, e.g. \\boxed{\\frac{1}{2}}.
    If there are several \\boxed{}, the last one is used (usually the final answer).

    Args:
        text: text containing \\boxed{}

    Returns:
        the extracted answer string, or None if not found
    """
    pattern = r'\\boxed\s*\{'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    # Take the last \boxed{} (usually the final answer)
    last_match = matches[-1]
    start = last_match.end()  # position right after the {

    # Stack matching: find the matching closing }
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == '{':
            depth += 1
        elif text[pos] == '}':
            depth -= 1
        pos += 1

    if depth != 0:
        return None

    # Extract the content inside {}
    answer = text[start:pos - 1].strip()
    return answer if answer else None


def extract_final_answer_heuristic(text: str, domain: str) -> str | None:
    """Heuristically extract the final answer (fallback when \\boxed{} is absent)

    Args:
        text: model output text
        domain: data domain (math/science)

    Returns:
        the extracted answer string, or None if not found
    """
    # Try to match the "the answer is X" pattern
    answer_patterns = [
        r'(?:the\s+)?answer\s+is\s*[:\s]*(.+?)(?:\.|$)',
        r'(?:the\s+)?result\s+is\s*[:\s]*(.+?)(?:\.|$)',
        r'(?:therefore|thus|hence|so)\s*,?\s*(.+?)(?:\.|$)',
    ]
    for pat in answer_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # science domain: try to match a multiple-choice answer (A/B/C/D/E)
    if domain == "science":
        m = re.search(r'\b([A-E])\b\s*$', text.strip())
        if m:
            return m.group(1).upper()

    return None


# ==================== Format conversion ====================

def convert_to_sharegpt_think(input_text: str, output_text: str) -> dict:
    """Convert one sample to the ShareGPT /think format

    Appends the /think control token to the end of the user message.

    Args:
        input_text: user question
        output_text: model answer (including the <think> reasoning)

    Returns:
        a dict in ShareGPT format
    """
    return {
        "conversations": [
            {"from": "user", "value": f"{input_text}/think"},
            {"from": "assistant", "value": output_text},
        ]
    }


def convert_to_sharegpt_no_think(input_text: str, output_text: str) -> dict:
    """Convert one sample to the ShareGPT /no_think format

    Appends the /no_think control token to the end of the user message.

    Args:
        input_text: user question
        output_text: model answer (short and direct)

    Returns:
        a dict in ShareGPT format
    """
    return {
        "conversations": [
            {"from": "user", "value": f"{input_text}/no_think"},
            {"from": "assistant", "value": output_text},
        ]
    }


def extract_ground_truth(output_text: str, domain: str) -> str | None:
    """Extract the ground-truth answer from the model output

    Prefers \\boxed{} extraction and falls back to the heuristic method.

    Args:
        output_text: model output text
        domain: data domain

    Returns:
        the ground-truth string, or None if it cannot be extracted
    """
    answer = extract_boxed_answer(output_text)
    if answer is not None:
        return answer
    return extract_final_answer_heuristic(output_text, domain)


def extract_after_think_block(output_text: str) -> str | None:
    """Extract the text after </think> from the output

    Used for the instruction-following domain: the content after the think block
    is taken directly as the /no_think answer.

    Args:
        output_text: model output text (may contain a <think>...</think> block)

    Returns:
        the text after </think>; the whole text if there is no think block;
        None if the extracted result is empty
    """
    idx = output_text.find("</think>")
    if idx == -1:
        # No think block: the whole text is the answer
        return output_text.strip() or None
    result = output_text[idx + len("</think>"):].strip()
    return result or None


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Convert the Superior-Reasoning-SFT dataset to the /think training format"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/superior_reasoning",
        help="Output directory (default: datasets/superior_reasoning)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="stage1",
        help="Training stages to include, comma-separated (default: stage1)",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="math,science",
        help="Domains to include, comma-separated (default: math,science)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace dataset cache directory",
    )
    args = parser.parse_args()

    # Parse arguments
    target_stages = set(args.stages.split(","))
    target_domains = set(args.domains.split(","))
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    think_output = output_dir / "think_data.jsonl"
    metadata_output = output_dir / "metadata.jsonl"
    instruction_no_think_output = output_dir / "no_think_instruction.jsonl"

    print(f"\n{'='*60}")
    print("Step 1: Prepare /think training data")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Target stages: {target_stages}")
    print(f"Target domains: {target_domains}")
    print(f"Output directory: {output_dir}")

    # ---- 1. Load the dataset ----
    print(f"\n{'─'*40}")
    print("1. Loading dataset from HuggingFace...")
    from datasets import load_dataset

    # Load several stages and merge
    all_data = []
    for stage in target_stages:
        print(f"  Loading {stage}...")
        ds = load_dataset(args.dataset_name, stage, cache_dir=args.cache_dir)
        # The dataset may have a train split or be a plain Dataset
        if isinstance(ds, dict):
            if "train" in ds:
                stage_data = ds["train"]
            else:
                first_key = list(ds.keys())[0]
                stage_data = ds[first_key]
        else:
            stage_data = ds

        print(f"    {stage} samples: {len(stage_data)}")
        all_data.extend(stage_data)

    data = all_data
    print(f"  Total samples: {len(data)}")

    # ---- 2. Filter by domain ----
    print(f"\n{'─'*40}")
    print("2. Filtering domains...")

    # Count per domain
    domain_counts = {}
    for item in data:
        d = item.get("domain", "unknown")
        domain_counts[d] = domain_counts.get(d, 0) + 1
    print(f"  Original domain distribution: {domain_counts}")

    # Filter (stages were already filtered at load time)
    filtered = []
    for item in data:
        domain = item.get("domain", "unknown")
        if domain in target_domains:
            filtered.append(item)

    print(f"  Samples after filtering: {len(filtered)}")

    # ---- 3. Convert format and extract ground truth ----
    print(f"\n{'─'*40}")
    print("3. Converting format and extracting ground truth...")

    think_count = 0
    gt_found_count = 0
    gt_missing_count = 0
    instruction_no_think_count = 0
    instruction_extract_fail_count = 0
    metadata_count = 0
    domain_stats = {}

    with open(think_output, "w", encoding="utf-8") as f_think, \
         open(metadata_output, "w", encoding="utf-8") as f_meta, \
         open(instruction_no_think_output, "w", encoding="utf-8") as f_inst:

        for item in filtered:
            uuid = item.get("uuid", "")
            input_text = item.get("input", "")
            output_text = item.get("output", "")
            domain = item.get("domain", "unknown")

            # The meta field is a JSON string and must be parsed
            meta_str = item.get("meta", "{}")
            try:
                meta_dict = json.loads(meta_str) if isinstance(meta_str, str) else meta_str
                stage = meta_dict.get("training_stage", "unknown")
            except (json.JSONDecodeError, AttributeError):
                stage = "unknown"

            # Skip empty samples
            if not input_text.strip() or not output_text.strip():
                continue

            # Initialize per-domain statistics
            if domain not in domain_stats:
                domain_stats[domain] = {"total": 0, "gt_found": 0, "no_think_extracted": 0}
            domain_stats[domain]["total"] += 1

            # Every domain is converted to the ShareGPT /think format
            sharegpt = convert_to_sharegpt_think(input_text, output_text)
            f_think.write(json.dumps(sharegpt, ensure_ascii=False) + "\n")
            think_count += 1

            if domain == "instruction-following":
                # instruction-following: take the text after </think> directly as /no_think
                no_think_text = extract_after_think_block(output_text)
                if no_think_text:
                    no_think_sharegpt = convert_to_sharegpt_no_think(input_text, no_think_text)
                    # Also save uuid and domain for the Step 3 filtering
                    no_think_item = {
                        "uuid": uuid,
                        "domain": domain,
                        **no_think_sharegpt,
                    }
                    f_inst.write(json.dumps(no_think_item, ensure_ascii=False) + "\n")
                    instruction_no_think_count += 1
                    domain_stats[domain]["no_think_extracted"] += 1
                else:
                    instruction_extract_fail_count += 1
            else:
                # math + science: extract ground truth and write it to metadata
                gt = extract_ground_truth(output_text, domain)
                if gt is not None:
                    gt_found_count += 1
                    domain_stats[domain]["gt_found"] = domain_stats[domain].get("gt_found", 0) + 1
                else:
                    gt_missing_count += 1

                meta = {
                    "uuid": uuid,
                    "input": input_text,
                    "domain": domain,
                    "ground_truth": gt,
                    "stage": stage,
                }
                f_meta.write(json.dumps(meta, ensure_ascii=False) + "\n")
                metadata_count += 1

            # Progress log
            if think_count % 10000 == 0:
                print(f"  Processed: {think_count}...")

    # ---- 4. Print statistics ----
    print(f"\n{'─'*40}")
    print("4. Results")
    print(f"  Total /think samples: {think_count}")
    print(f"  metadata samples (math+science): {metadata_count}")
    print(f"  Ground truth extracted: {gt_found_count} ({gt_found_count/max(metadata_count,1)*100:.1f}%)")
    print(f"  Ground truth missing: {gt_missing_count} ({gt_missing_count/max(metadata_count,1)*100:.1f}%)")
    print(f"  instruction /no_think extracted: {instruction_no_think_count}")
    print(f"  instruction /no_think failed: {instruction_extract_fail_count}")
    print(f"\n  Per-domain statistics:")
    for domain, stats in sorted(domain_stats.items()):
        total = stats["total"]
        if domain == "instruction-following":
            extracted = stats.get("no_think_extracted", 0)
            print(f"    {domain}: {total} items, /no_think extraction rate {extracted/max(total,1)*100:.1f}%")
        else:
            found = stats.get("gt_found", 0)
            print(f"    {domain}: {total} items, GT extraction rate {found/max(total,1)*100:.1f}%")

    print(f"\n  Output files:")
    print(f"    /think data:              {think_output}")
    print(f"    metadata (math+science):  {metadata_output}")
    print(f"    /no_think (instruction):  {instruction_no_think_output}")
    print(f"\n{'='*60}")
    print("Step 1 done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
