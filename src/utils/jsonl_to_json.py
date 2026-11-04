"""
JSONL to JSON Converter

Usage:
  # Take the first 100 records in order
  python jsonl_to_json.py data.jsonl 100

  # Take 100 records at random
  python jsonl_to_json.py data.jsonl 100 --shuffle
"""

import json
import argparse
import random
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Convert JSONL to pretty JSON file.")
    parser.add_argument("input", type=str, help="Path to the target .jsonl file")
    parser.add_argument("count", type=int, help="Number of records to convert")
    parser.add_argument("--shuffle", action="store_true", default=False,
                        help="Sample randomly (default: take the first records in order)")
    return parser.parse_args()


def main():
    args = parse_args()
    # Check that the file exists
    if not os.path.isfile(args.input):
        print(f"[Error] File not found: {args.input}")
        sys.exit(1)

    # Read all lines
    with open(args.input, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    total = len(lines)
    print(f"[Info] File has {total} records")

    if args.count > total:
        print(f"[Warning] Requested {args.count} records but the file has only {total}; exporting all")
        args.count = total

    # Select records
    if args.shuffle:
        selected_lines = random.sample(lines, args.count)
        print(f"[Info] Randomly selected {args.count} records")
    else:
        selected_lines = lines[:args.count]
        print(f"[Info] Selected the first {args.count} records in order")

    # Parse JSON
    records = []
    for i, line in enumerate(selected_lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[Warning] Record {i+1} failed to parse, skipped: {e}")

    # Build the output path (same name and directory, extension changed to .json)
    base, _ = os.path.splitext(args.input)
    output_path = base + ".json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[Done] Saved {len(records)} records to: {output_path}")


if __name__ == "__main__":
    main()