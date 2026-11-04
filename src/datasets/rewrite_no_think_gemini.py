#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rewrite self-reflective CoT answers into concise step-by-step solutions with the Gemini API

Takes the long self-reflective reasoning already present in the Superior Reasoning dataset
and feeds it to Gemini 3.1 Pro, which condenses it into a straight-forward step-by-step solution.

Rewriting an existing correct reasoning path (instead of generating from scratch) keeps the pass rate high.

Input:
  - HuggingFace: Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b (stage1)
  - metadata.jsonl (uuid -> ground_truth mapping)

Output:
  - no_think_gemini_rewrite.jsonl (format compatible with filter_no_think_data.py)

Usage:

    # Smoke test (20 samples)
    python -m src.datasets.rewrite_no_think_gemini --limit 20

    # Full run
    python -m src.datasets.rewrite_no_think_gemini

Dependencies:
    pip install google-genai datasets
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Please install google-genai: pip install google-genai")
    sys.exit(1)

# Project root directory
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ==================== Prompt ====================

SYSTEM_PROMPT = """You are a mathematics and science tutor who writes clean, step-by-step solutions.
You never use self-reflective language like "wait", "hmm", "actually", "let me reconsider", "alternatively", "on second thought", "I made an error", etc.
You write in a confident, forward-only, declarative style."""

REWRITE_PROMPT_TEMPLATE = """Below is a math/science question and a detailed (but verbose and self-reflective) solution.
Your task: rewrite the solution into a clean, concise step-by-step format.

Rules:
1. Extract ONLY the correct reasoning steps. Remove ALL self-reflection, backtracking, and hesitation.
2. Preserve all key intermediate steps — the reader should be able to follow the solution method.
3. Do NOT add any new reasoning or change the mathematical approach.
4. Do NOT use phrases like: "Wait", "Hmm", "Actually", "Let me reconsider", "Alternatively", "On second thought", "I think", "Let's", "Let me", "We need to", "Recall that", "Note that", "Consider", "Think", "Okay", "To solve this".
5. Write in a direct, declarative style: "The equation becomes...", "Substituting x=3...", "This gives...", etc.
6. Put the final answer in \\boxed{{}}.
7. Do NOT include <think> or </think> tags.
8. Use LaTeX notation for mathematical expressions.

**Question:**
{question}

**Verbose Solution:**
{solution}

**Clean Step-by-Step Solution:**"""


# ==================== Reflective-word detection (reuses the patterns of filter_no_think_data.py) ====================

REASONING_PATTERNS = [
    r'\bwait\b',
    r'\blet me think\b',
    r'\blet\'s\b',
    r'\bfirst\b.*\bthen\b',
    r'\bstep \d+\b',
    r'\bhmm+\b',
    r'\bokay\b',
    r'\bwe need to\b',
    r'\bwe can\b',
    r'\bto solve this\b',
    r'\brecall that\b',
    r'\bnote that\b',
    r'\bconsider\b',
    r'\bthink\b',
    r'<think>',
]
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in REASONING_PATTERNS]


def check_no_reasoning_style(text: str) -> bool:
    """Check whether the text contains reflective words; True = pass (no reflective words)"""
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(text):
            return False
    return True


# ==================== Answer extraction ====================

def extract_boxed_answer(text: str) -> str | None:
    """Extract the answer inside \\boxed{...} from text (stack matching, nested braces supported)"""
    pattern = r'\\boxed\s*\{'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    last_match = matches[-1]
    start = last_match.end()

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

    answer = text[start:pos - 1].strip()
    return answer if answer else None


# ==================== Resume from checkpoint ====================

def load_completed_uuids(output_path: str) -> set:
    """Scan the existing output file and collect the set of completed uuids"""
    completed = set()
    if not os.path.exists(output_path):
        return completed

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if "uuid" in item:
                    completed.add(item["uuid"])
            except json.JSONDecodeError:
                continue

    return completed


# ==================== Data loading ====================

def load_dataset_with_ground_truth(
    dataset_name: str,
    stages: list[str],
    domains: set[str],
    metadata_path: str,
    cache_dir: str | None = None,
) -> list[dict]:
    """Load the dataset from HuggingFace and attach the ground_truth from metadata

    Returns:
        list of samples, each with uuid, input, output, domain, ground_truth
    """
    from datasets import load_dataset

    # Load the HuggingFace dataset
    all_data = []
    for stage in stages:
        print(f"  Loading {stage}...")
        ds = load_dataset(dataset_name, stage, cache_dir=cache_dir)
        if isinstance(ds, dict):
            stage_data = ds.get("train", ds[list(ds.keys())[0]])
        else:
            stage_data = ds
        print(f"    {stage} samples: {len(stage_data)}")
        all_data.extend(stage_data)

    # Filter domains
    filtered = [item for item in all_data if item.get("domain", "") in domains]
    print(f"  After filtering ({', '.join(domains)}): {len(filtered)}")

    # Load the ground_truth mapping
    gt_map = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    meta = json.loads(line)
                    gt_map[meta.get("uuid", "")] = meta.get("ground_truth")
        print(f"  metadata ground_truth: {len(gt_map)}")

    # Merge
    result = []
    for item in filtered:
        uuid = item.get("uuid", "")
        result.append({
            "uuid": uuid,
            "input": item.get("input", ""),
            "output": item.get("output", ""),
            "domain": item.get("domain", ""),
            "ground_truth": gt_map.get(uuid),
        })

    return result


# ==================== Gemini API call ====================

async def call_gemini(
    client: genai.Client,
    model_name: str,
    question: str,
    solution: str,
    max_output_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    rpm_delay: float,
) -> tuple[str | None, dict | None]:
    """Call the Gemini API to rewrite one sample

    Returns:
        (rewritten result, error info)
        success: (text, None)
        failure: (None, {"error_type": ..., "error_message": ...})
    """
    prompt = REWRITE_PROMPT_TEMPLATE.format(question=question, solution=solution)

    last_error = None
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                    ),
                )

                # Extract the text and completion status
                text = None
                finish_reason = None
                try:
                    text = response.text
                except Exception:
                    # response.text may raise when the response is blocked
                    pass

                # Get finish_reason
                if response.candidates:
                    finish_reason = response.candidates[0].finish_reason

                if text:
                    # RPM rate limiting
                    await asyncio.sleep(rpm_delay)
                    # Check for truncation
                    if finish_reason and str(finish_reason) not in ("STOP", "FinishReason.STOP", "0"):
                        return (text, {
                            "error_type": "truncated",
                            "error_message": f"finish_reason={finish_reason}",
                            "details": {"text_length": len(text)},
                        })
                    return (text, None)
                else:
                    last_error = {
                        "error_type": "empty_response",
                        "error_message": f"Gemini returned empty text, finish_reason={finish_reason}",
                        "details": {"attempts": attempt + 1},
                    }
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return (None, last_error)

            except Exception as e:
                error_name = type(e).__name__
                last_error = {
                    "error_type": error_name,
                    "error_message": str(e)[:500],
                    "details": {"attempts": attempt + 1},
                }
                # Rate-limit errors need a longer wait
                if "ResourceExhausted" in error_name or "429" in str(e):
                    wait_time = 30 * (attempt + 1)
                    print(f"  Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                elif attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return (None, last_error)

    return (None, last_error)


# ==================== Batch processing ====================

async def rewrite_batch(
    items: list[dict],
    client: genai.Client,
    model_name: str,
    max_output_tokens: int,
    temperature: float,
    batch_size: int,
    rpm_limit: int,
    output_path: str,
):
    """Rewrite in asynchronous batches"""
    semaphore = asyncio.Semaphore(batch_size)
    rpm_delay = 60.0 / rpm_limit if rpm_limit > 0 else 0

    completed = 0
    failed = 0
    style_flagged = 0
    start_time = time.time()

    f_out = open(output_path, "a", encoding="utf-8")

    async def process_item(item):
        nonlocal completed, failed, style_flagged

        uuid = item["uuid"]
        question = item["input"]
        solution = item["output"]
        domain = item["domain"]
        ground_truth = item["ground_truth"]

        result, error_info = await call_gemini(
            client, model_name, question, solution,
            max_output_tokens, temperature, semaphore, rpm_delay,
        )

        if result is not None:
            # Extract the answer
            extracted_answer = extract_boxed_answer(result)

            # Inline style check
            style_ok = check_no_reasoning_style(result)
            if not style_ok:
                style_flagged += 1

            output_item = {
                "uuid": uuid,
                "input": question,
                "domain": domain,
                "generated_output": result,
                "extracted_answer": extracted_answer,
                "ground_truth": ground_truth,
                "style_check_passed": style_ok,
                "original_output_length": len(solution),
            }
            f_out.write(json.dumps(output_item, ensure_ascii=False) + "\n")
            f_out.flush()
            completed += 1
        else:
            failed += 1
            # Record failed samples
            error_item = {
                "uuid": uuid,
                "input": question,
                "domain": domain,
                "generated_output": None,
                "ground_truth": ground_truth,
                "error": error_info,
            }
            f_out.write(json.dumps(error_item, ensure_ascii=False) + "\n")
            f_out.flush()

        # Progress log
        total_done = completed + failed
        if total_done % 100 == 0 and total_done > 0:
            elapsed = time.time() - start_time
            speed = total_done / elapsed
            remaining = (len(items) - total_done) / max(speed, 0.01)
            print(
                f"  Progress: {total_done}/{len(items)} "
                f"({total_done/len(items)*100:.1f}%) | "
                f"Succeeded: {completed} | Failed: {failed} | "
                f"Style flagged: {style_flagged} | "
                f"Speed: {speed:.1f} items/s | "
                f"ETA: {remaining/60:.1f} min"
            )

    # Run concurrently
    tasks = [process_item(item) for item in items]
    await asyncio.gather(*tasks)

    f_out.close()

    # Final statistics
    elapsed = time.time() - start_time
    print(f"\n  Rewrite finished!")
    print(f"  Succeeded: {completed} | Failed: {failed} | Style flagged: {style_flagged}")
    print(f"  Style pass rate: {(completed - style_flagged) / max(completed, 1) * 100:.1f}%")
    print(f"  Total time: {elapsed / 60:.1f} min")
    print(f"  Average speed: {(completed + failed) / max(elapsed, 0.01):.1f} items/s")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Rewrite self-reflective CoT answers into concise step-by-step solutions with the Gemini API"
    )
    parser.add_argument(
        "--output", type=str,
        default="datasets/superior_reasoning/no_think_gemini_rewrite.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--metadata_input", type=str,
        default="datasets/superior_reasoning/metadata.jsonl",
        help="Path to metadata.jsonl (for ground_truth)",
    )
    parser.add_argument(
        "--dataset_name", type=str,
        default="Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b",
        help="HuggingFace dataset name",
    )
    parser.add_argument("--stages", type=str, default="stage1", help="Dataset stages")
    parser.add_argument("--domains", type=str, default="math,science", help="Domains to keep")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N samples (0 = no limit, for smoke tests)")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of concurrent requests")
    parser.add_argument("--rpm_limit", type=int, default=60, help="Maximum requests per minute")
    parser.add_argument("--max_output_tokens", type=int, default=4096, help="Maximum number of output tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--model_name", type=str, default="gemini-3.1-pro-preview", help="Gemini model name")
    parser.add_argument("--api_key_env", type=str, default="GEMINI_API_KEY", help="Environment variable holding the API key")
    parser.add_argument("--cache_dir", type=str, default=None, help="HuggingFace cache directory")
    args = parser.parse_args()

    # Resolve paths
    output_path = project_root / args.output
    metadata_path = project_root / args.metadata_input
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_domains = set(args.domains.split(","))
    target_stages = args.stages.split(",")

    # Get the API key
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"ERROR: please set the environment variable {args.api_key_env}")
        print(f"  export {args.api_key_env}='your-api-key'")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Gemini Rewrite: self-reflective CoT -> step-by-step solution")
    print(f"{'='*60}")
    print(f"Model:       {args.model_name}")
    print(f"Output:      {output_path}")
    print(f"Concurrency: {args.batch_size}")
    print(f"RPM limit:   {args.rpm_limit}")
    print(f"max_tokens: {args.max_output_tokens}")
    print(f"temperature: {args.temperature}")
    if args.limit > 0:
        print(f"Limit:       only the first {args.limit} samples")

    # ---- 1. Resume check ----
    print(f"\n{'─'*40}")
    print("1. Checking resume state...")
    completed_uuids = load_completed_uuids(str(output_path))
    if completed_uuids:
        print(f"  Found {len(completed_uuids)} completed records, they will be skipped")
    else:
        print(f"  No completed records, starting from scratch")

    # ---- 2. Load the dataset ----
    print(f"\n{'─'*40}")
    print("2. Loading dataset...")
    all_items = load_dataset_with_ground_truth(
        dataset_name=args.dataset_name,
        stages=target_stages,
        domains=target_domains,
        metadata_path=str(metadata_path),
        cache_dir=args.cache_dir,
    )

    # Filter out completed items
    pending = [item for item in all_items if item["uuid"] not in completed_uuids]
    print(f"  Total samples: {len(all_items)}")
    print(f"  Pending:       {len(pending)}")
    print(f"  Skipped:       {len(all_items) - len(pending)}")

    # Apply --limit
    if args.limit > 0 and len(pending) > args.limit:
        pending = pending[:args.limit]
        print(f"  Applying --limit={args.limit}, actually processing: {len(pending)}")

    if not pending:
        print("\n  All samples have been processed.")
        return

    # ---- 3. Initialize the Gemini client ----
    print(f"\n{'─'*40}")
    print("3. Initializing Gemini client...")
    client = genai.Client(api_key=api_key)

    # Test connectivity
    try:
        test_response = client.models.generate_content(
            model=args.model_name,
            contents="Say 'hello' in one word.",
            config=types.GenerateContentConfig(max_output_tokens=10),
        )
        test_text = ""
        try:
            test_text = test_response.text or ""
        except Exception:
            test_text = str(test_response.candidates[0].content.parts[0].text) if test_response.candidates else ""
        print(f"  API connection OK: {test_text[:50]}")
    except Exception as e:
        print(f"  WARNING: API connectivity test failed: {e}")
        print(f"  Will keep trying...")

    # ---- 4. Start rewriting ----
    print(f"\n{'─'*40}")
    print("4. Starting batch rewrite...")
    asyncio.run(
        rewrite_batch(
            items=pending,
            client=client,
            model_name=args.model_name,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
            rpm_limit=args.rpm_limit,
            output_path=str(output_path),
        )
    )

    # ---- 5. Statistics ----
    print(f"\n{'─'*40}")
    print("5. Output statistics...")
    total_out = 0
    success_out = 0
    style_pass = 0
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            total_out += 1
            if item.get("generated_output") is not None:
                success_out += 1
                if item.get("style_check_passed", False):
                    style_pass += 1

    print(f"  Total lines in output file: {total_out}")
    print(f"  Successfully generated: {success_out}")
    print(f"  Style passed: {style_pass} ({style_pass / max(success_out, 1) * 100:.1f}%)")
    print(f"  Output file: {output_path}")

    print(f"\n{'='*60}")
    print("Rewrite finished!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
