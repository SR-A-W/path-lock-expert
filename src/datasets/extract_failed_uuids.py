#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract UUIDs of failed samples

Two modes:
1. Default mode: extract the UUIDs of all failed samples from a *_failed.json file
2. Compare mode: compare the original dataset (metadata.jsonl) with the generated dataset (no_think_raw.jsonl)
   and find the samples that were not generated successfully

Usage:

    # Default mode: extract from *_failed.json
    python -m src.datasets.extract_failed_uuids \
        --failed_json datasets/superior_reasoning/no_think_raw_failed.json

    # Specify the output file name
    python -m src.datasets.extract_failed_uuids \
        --failed_json datasets/superior_reasoning/no_think_raw_failed.json \
        --output datasets/superior_reasoning/rejected_uuids.jsonl

    # Compare mode: compare the original and generated datasets
    python -m src.datasets.extract_failed_uuids \
        --compare \
        --metadata datasets/superior_reasoning/metadata.jsonl \
        --generated datasets/superior_reasoning/no_think_raw.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Project root directory
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_uuids_from_jsonl(file_path: Path) -> set:
    """Load all UUIDs from a JSONL file

    Args:
        file_path: JSONL file path

    Returns:
        set of UUIDs
    """
    uuids = set()
    if not file_path.exists():
        print(f"Warning: file not found: {file_path}")
        return uuids

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if "uuid" in item:
                    uuids.add(item["uuid"])
            except json.JSONDecodeError as e:
                print(f"Warning: JSON parse failed at line {line_num}: {e}")
                continue

    return uuids


def load_metadata_with_domain(file_path: Path) -> dict:
    """Load UUIDs and their domains from metadata.jsonl

    Args:
        file_path: path to metadata.jsonl

    Returns:
        a {uuid: {"domain": domain, "input": input_text}} dict
    """
    uuid_info = {}
    if not file_path.exists():
        print(f"Error: file not found: {file_path}")
        return uuid_info

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                uuid = item.get("uuid")
                if uuid:
                    uuid_info[uuid] = {
                        "domain": item.get("domain", "unknown"),
                        "input": item.get("input", ""),
                    }
            except json.JSONDecodeError as e:
                print(f"Warning: JSON parse failed at line {line_num}: {e}")
                continue

    return uuid_info


def load_failed_samples_from_json(file_path: Path) -> list:
    """Load failed samples from a *_failed.json file

    Args:
        file_path: path to the *_failed.json file

    Returns:
        list of failed samples, each with uuid, domain, reason, input, etc.
    """
    if not file_path.exists():
        print(f"Error: file not found: {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            failed_samples = data.get("failed_samples", [])
            return failed_samples
    except json.JSONDecodeError as e:
        print(f"Error: JSON parse failed: {e}")
        return []
    except Exception as e:
        print(f"Error: failed to read file: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Extract UUIDs of failed samples (default: from *_failed.json, or by comparing the original and generated datasets)"
    )

    # Mode selection
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Use compare mode (compare the metadata and generated files)",
    )

    # Default-mode arguments
    parser.add_argument(
        "--failed_json",
        type=str,
        default=None,
        help="Path to the failed-samples JSON file (*_failed.json)",
    )

    # Compare-mode arguments
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Original dataset path (metadata.jsonl, compare mode only)",
    )
    parser.add_argument(
        "--generated",
        type=str,
        default=None,
        help="Generated dataset path (no_think_raw.jsonl, compare mode only)",
    )

    # Common arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: derived from the input file)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "jsonl"],
        default="jsonl",
        help="Output format: json (a single JSON array) or jsonl (one JSON object per line, default)",
    )
    args = parser.parse_args()

    # Argument validation
    if args.compare:
        # Compare mode
        if not args.metadata or not args.generated:
            print("Error: compare mode requires both --metadata and --generated")
            return

        metadata_path = project_root / args.metadata
        generated_path = project_root / args.generated

        # Output path: default is the generated file name + _rejected_uuid.jsonl
        if args.output:
            output_path = project_root / args.output
        else:
            suffix = ".json" if args.format == "json" else ".jsonl"
            output_path = generated_path.parent / (generated_path.stem + "_rejected_uuid" + suffix)

        run_compare_mode(metadata_path, generated_path, output_path, args.format)
    else:
        # Default mode: extract from *_failed.json
        if not args.failed_json:
            print("Error: default mode requires --failed_json")
            print("Hint: use --compare to switch to compare mode")
            return

        failed_json_path = project_root / args.failed_json

        # Output path: default is rejected_uuids.jsonl in the same directory
        if args.output:
            output_path = project_root / args.output
        else:
            output_path = failed_json_path.parent / "rejected_uuids.jsonl"

        run_default_mode(failed_json_path, output_path, args.format)


def run_default_mode(failed_json_path: Path, output_path: Path, output_format: str):
    """Default mode: extract failed-sample UUIDs from *_failed.json

    Args:
        failed_json_path: path to the *_failed.json file
        output_path: output file path
        output_format: output format (json or jsonl)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Extracting failed-sample UUIDs (default mode)")
    print(f"{'='*60}")
    print(f"Failed-samples file: {failed_json_path}")
    print(f"Output file:         {output_path}")
    print(f"Output format:       {output_format}")

    # ---- 1. Load failed samples ----
    print(f"\n{'─'*40}")
    print("1. Loading failed samples...")

    failed_samples = load_failed_samples_from_json(failed_json_path)
    print(f"  Number of failed samples: {len(failed_samples)}")

    if not failed_samples:
        print("\n  No failed samples found.")
        return

    # ---- 2. Count by domain and error_type ----
    print(f"\n{'─'*40}")
    print("2. Counting failed samples...")

    domain_stats = {}
    error_type_stats = {}

    for sample in failed_samples:
        domain = sample.get("domain", "unknown")
        error_type = sample.get("error_type", "unknown")

        domain_stats[domain] = domain_stats.get(domain, 0) + 1
        error_type_stats[error_type] = error_type_stats.get(error_type, 0) + 1

    print(f"\n  Failed samples by domain:")
    for domain, count in sorted(domain_stats.items()):
        print(f"    {domain}: {count}")

    print(f"\n  Failed samples by error_type:")
    for error_type, count in sorted(error_type_stats.items()):
        print(f"    {error_type}: {count}")

    # ---- 3. Prepare output data ----
    print(f"\n{'─'*40}")
    print("3. Preparing output data...")

    output_items = []
    for sample in failed_samples:
        output_items.append({
            "uuid": sample.get("uuid", ""),
            "domain": sample.get("domain", "unknown"),
            "reason": f"generation_failed_{sample.get('error_type', 'unknown')}",
            "input": sample.get("input", ""),
        })

    # ---- 4. Write the output file ----
    print(f"\n{'─'*40}")
    print("4. Writing output file...")

    if output_format == "json":
        # JSON format: a single array
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_items, f, ensure_ascii=False, indent=2)
    else:
        # JSONL format: one object per line
        with open(output_path, "w", encoding="utf-8") as f:
            for item in output_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  Wrote {len(output_items)} failed samples")
    print(f"  Output file: {output_path}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


def run_compare_mode(metadata_path: Path, generated_path: Path, output_path: Path, output_format: str):
    """Compare mode: compare the original and generated datasets to find failed samples

    Args:
        metadata_path: path to metadata.jsonl
        generated_path: path to the generated dataset file
        output_path: output file path
        output_format: output format (json or jsonl)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Extracting failed-sample UUIDs (compare mode)")
    print(f"{'='*60}")
    print(f"Original dataset:  {metadata_path}")
    print(f"Generated dataset: {generated_path}")
    print(f"Output file:       {output_path}")
    print(f"Output format:     {output_format}")

    # ---- 1. Load UUIDs and domains of the original dataset ----
    print(f"\n{'─'*40}")
    print("1. Loading original dataset...")

    metadata_info = load_metadata_with_domain(metadata_path)
    print(f"  Original dataset UUIDs: {len(metadata_info)}")

    if not metadata_info:
        print("Error: original dataset is empty or unreadable")
        return

    # ---- 2. Load UUIDs of the generated dataset ----
    print(f"\n{'─'*40}")
    print("2. Loading generated dataset...")

    generated_uuids = load_uuids_from_jsonl(generated_path)
    print(f"  Generated dataset UUIDs: {len(generated_uuids)}")

    # ---- 3. Find the failed UUIDs ----
    print(f"\n{'─'*40}")
    print("3. Comparing and extracting failed samples...")

    failed_uuids = set(metadata_info.keys()) - generated_uuids
    print(f"  Number of failed samples: {len(failed_uuids)}")

    if not failed_uuids:
        print("\n  All samples were generated successfully; no failed samples.")
        return

    # ---- 4. Count by domain ----
    domain_stats = {}
    failed_items = []

    for uuid in failed_uuids:
        info = metadata_info[uuid]
        domain = info["domain"]
        domain_stats[domain] = domain_stats.get(domain, 0) + 1

        failed_items.append({
            "uuid": uuid,
            "domain": domain,
            "reason": "not_generated",
            "input": info["input"],
        })

    print(f"\n  Failed samples by domain:")
    for domain, count in sorted(domain_stats.items()):
        print(f"    {domain}: {count}")

    # ---- 5. Write the output file ----
    print(f"\n{'─'*40}")
    print("4. Writing output file...")

    if output_format == "json":
        # JSON format: a single array
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(failed_items, f, ensure_ascii=False, indent=2)
    else:
        # JSONL format: one object per line
        with open(output_path, "w", encoding="utf-8") as f:
            for item in failed_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  Wrote {len(failed_items)} failed samples")
    print(f"  Output file: {output_path}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
