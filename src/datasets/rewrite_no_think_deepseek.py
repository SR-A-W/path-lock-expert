#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rewrite self-reflective CoT answers into concise step-by-step solutions with the DeepSeek API

Takes the long self-reflective reasoning already present in the Superior Reasoning dataset
and feeds it to DeepSeek V3.2, which condenses it into a straight-forward step-by-step solution.

Rewriting an existing correct reasoning path (instead of generating from scratch) keeps the pass rate high.

Input:
  - HuggingFace: Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b (stage1)
  - metadata.jsonl (uuid -> ground_truth mapping)

Output:
  - no_think_deepseek_rewrite.jsonl (format compatible with filter_no_think_data.py)

Usage:

    # Smoke test (20 samples)
    python -m src.datasets.rewrite_no_think_deepseek --limit 20

    # Full run
    python -m src.datasets.rewrite_no_think_deepseek

Dependencies:
    pip install aiohttp datasets
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
    import aiohttp
except ImportError:
    print("Please install aiohttp: pip install aiohttp")
    sys.exit(1)

# Project root directory
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ==================== Prompt ====================

SYSTEM_PROMPT = """You are a mathematics and science tutor who writes detailed, clean, step-by-step solutions for students.
You never use self-reflective language like "wait", "hmm", "actually", "let me reconsider", "alternatively", "on second thought", "I made an error", etc.
You write in a confident, forward-only, declarative style. Every step is explained clearly so that a student can follow the reasoning."""

REWRITE_PROMPT_TEMPLATE = """Below is a math/science question and a detailed (but verbose and self-reflective) solution.
Your task: rewrite the solution into a clean, concise step-by-step format.

Rules:
1. Remove ALL self-reflection, backtracking, and hesitation from the original solution.
2. **Keep it concise but complete**: include the key reasoning steps and important intermediate results. Merge trivial algebraic steps (e.g., "multiply both sides by 2" and "divide by n" can be one step). Do NOT split simple operations into separate steps.
3. **Explain non-obvious steps only**: when a non-trivial technique is used (e.g., AM-GM inequality, substitution, change of variables), add a brief one-sentence explanation of why it works. Routine algebra does not need explanation.
4. Do NOT add reasoning that wasn't in the original solution. Keep the same mathematical approach.
5. Do NOT use self-reflective phrases: "Wait", "Hmm", "Actually", "Let me reconsider", "Alternatively", "On second thought", "I think", "Let's", "Let me", "We need to", "Recall that", "Note that", "Consider", "Think", "Okay", "To solve this".
6. Write in a direct, declarative style: "The equation becomes...", "Substituting x=3 gives...", "This simplifies to...", "By the AM-GM inequality...", etc. You may use "Step 1/2/3" labels to organize the solution if helpful.
7. Put the final answer in \\boxed{{}}.
8. Do NOT include <think> or </think> tags.
9. Use LaTeX notation for mathematical expressions.
10. Aim for a solution length roughly between 300-1500 characters. Shorter for easy problems, longer for hard ones.

**Question:**
{question}

**Verbose Solution:**
{solution}

**Clean Solution:**"""


# ==================== Reflective-word detection ====================

REASONING_PATTERNS = [
    r'\bwait\b',
    r'\blet me think\b',
    r'\blet\'s\b',
    r'\bhmm+\b',
    r'\bokay\b',
    r'\bto solve this\b',
    r'\bthink\b',
    r'<think>',
]
# The patterns below are NOT used as filters (normal math language, not self-reflective):
# r'\bstep \d+\b'  — structured step markers
# r'\bconsider\b'  — normal wording in math (e.g., "consider the function f(x)")
# r'\bwe can\b', r'\bwe need to\b' — normal problem-solving language
# r'\bnote that\b', r'\brecall that\b' — citing theorems/properties
# r'\bfirst\b.*\bthen\b' — description of step order
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
    """Load the dataset from HuggingFace and attach the ground_truth from metadata"""
    from datasets import load_dataset

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

    filtered = [item for item in all_data if item.get("domain", "") in domains]
    print(f"  After filtering ({', '.join(domains)}): {len(filtered)}")

    gt_map = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    meta = json.loads(line)
                    gt_map[meta.get("uuid", "")] = meta.get("ground_truth")
        print(f"  metadata ground_truth: {len(gt_map)}")

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


# ==================== DeepSeek API call ====================

async def call_deepseek(
    session: aiohttp.ClientSession,
    api_url: str,
    api_key: str,
    model_name: str,
    question: str,
    solution: str,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    rpm_delay: float,
) -> tuple[str | None, dict | None]:
    """Call the DeepSeek OpenAI-compatible API to rewrite one sample

    Returns:
        (rewritten result, error info)
        success: (text, None)
        failure: (None, {"error_type": ..., "error_message": ...})
    """
    prompt = REWRITE_PROMPT_TEMPLATE.format(question=question, solution=solution)

    url = f"{api_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # deepseek-reasoner does not support sampling parameters such as temperature/top_p
    is_reasoner = "reasoner" in model_name.lower()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    # Only non-reasoner mode supports temperature
    if not is_reasoner:
        payload["temperature"] = temperature

    last_error = None
    async with semaphore:
        for attempt in range(3):
            try:
                async with session.post(
                    url, json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        message = data["choices"][0]["message"]
                        # deepseek-reasoner returns reasoning_content (CoT) + content (final answer)
                        # We only take content (the final clean answer)
                        content = message.get("content", "")
                        finish_reason = data["choices"][0].get("finish_reason", "")

                        if content:
                            await asyncio.sleep(rpm_delay)
                            if finish_reason == "length":
                                return (content, {
                                    "error_type": "truncated",
                                    "error_message": f"finish_reason=length",
                                    "details": {"text_length": len(content)},
                                })
                            return (content, None)
                        else:
                            last_error = {
                                "error_type": "empty_response",
                                "error_message": f"API returned empty content, finish_reason={finish_reason}",
                                "details": {"attempts": attempt + 1},
                            }
                            if attempt < 2:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return (None, last_error)

                    elif resp.status == 429:
                        # Rate limiting
                        wait_time = 30 * (attempt + 1)
                        print(f"  Rate limited (429), waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        last_error = {
                            "error_type": "rate_limit",
                            "error_message": f"HTTP 429",
                            "details": {"attempts": attempt + 1},
                        }
                        continue

                    else:
                        error_text = await resp.text()
                        last_error = {
                            "error_type": "api_error",
                            "error_message": f"HTTP {resp.status}: {error_text[:300]}",
                            "details": {"attempts": attempt + 1, "status_code": resp.status},
                        }
                        if attempt < 2:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return (None, last_error)

            except asyncio.TimeoutError:
                last_error = {
                    "error_type": "timeout",
                    "error_message": "Request timed out (300s)",
                    "details": {"attempts": attempt + 1},
                }
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return (None, last_error)

            except aiohttp.ClientError as e:
                last_error = {
                    "error_type": "network_error",
                    "error_message": f"{type(e).__name__}: {str(e)[:300]}",
                    "details": {"attempts": attempt + 1},
                }
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return (None, last_error)

    return (None, last_error)


# ==================== Batch processing ====================

async def rewrite_batch(
    items: list[dict],
    api_url: str,
    api_key: str,
    model_name: str,
    max_tokens: int,
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
    per_item_times = []
    start_time = time.time()

    # Print progress for every item at small scale, every 100 items at large scale
    log_interval = 1 if len(items) <= 50 else 100

    f_out = open(output_path, "a", encoding="utf-8")

    async with aiohttp.ClientSession() as session:

        async def process_item(item):
            nonlocal completed, failed, style_flagged

            uuid = item["uuid"]
            question = item["input"]
            solution = item["output"]
            domain = item["domain"]
            ground_truth = item["ground_truth"]

            item_start = time.time()
            result, error_info = await call_deepseek(
                session, api_url, api_key, model_name,
                question, solution, max_tokens, temperature,
                semaphore, rpm_delay,
            )
            item_elapsed = time.time() - item_start
            per_item_times.append(item_elapsed)

            if result is not None:
                extracted_answer = extract_boxed_answer(result)
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

            total_done = completed + failed
            if total_done % log_interval == 0 and total_done > 0:
                elapsed = time.time() - start_time
                speed = total_done / elapsed
                remaining = (len(items) - total_done) / max(speed, 0.01)
                print(
                    f"  [{total_done}/{len(items)}] "
                    f"Succeeded:{completed} Failed:{failed} Style flagged:{style_flagged} | "
                    f"This item:{item_elapsed:.1f}s | "
                    f"Average:{sum(per_item_times)/len(per_item_times):.1f}s/item | "
                    f"Speed:{speed:.2f} items/s | "
                    f"Remaining:{remaining/60:.1f} min"
                )

        tasks = [process_item(item) for item in items]
        await asyncio.gather(*tasks)

    f_out.close()

    elapsed = time.time() - start_time
    avg_time = sum(per_item_times) / len(per_item_times) if per_item_times else 0
    print(f"\n  {'='*50}")
    print(f"  Rewrite finished!")
    print(f"  {'='*50}")
    print(f"  Succeeded: {completed} | Failed: {failed} | Style flagged: {style_flagged}")
    print(f"  Style pass rate: {(completed - style_flagged) / max(completed, 1) * 100:.1f}%")
    print(f"  Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Average per item: {avg_time:.1f}s")
    print(f"  Throughput: {(completed + failed) / max(elapsed, 0.01):.2f} items/s")
    print(f"  {'─'*50}")
    print(f"  Full-run estimate (65K samples):")
    print(f"    Time: {65150 * avg_time / 3600:.1f} hours (serial)")
    print(f"    Time: {65150 / max((completed + failed) / max(elapsed, 0.01), 0.01) / 3600:.1f} hours (current concurrency)")
    print(f"  {'='*50}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Rewrite self-reflective CoT answers into concise step-by-step solutions with the DeepSeek API"
    )
    parser.add_argument(
        "--output", type=str,
        default="datasets/hybrid_superior_reasoning/no_think_deepseek_rewrite.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--metadata_input", type=str,
        default="datasets/superior_reasoning/metadata.jsonl",
        help="Path to metadata.jsonl (for ground_truth, still read from superior_reasoning)",
    )
    parser.add_argument(
        "--dataset_name", type=str,
        default="Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b",
        help="HuggingFace dataset name",
    )
    parser.add_argument("--stages", type=str, default="stage1", help="Dataset stages")
    parser.add_argument("--domains", type=str, default="math,science", help="Domains to keep")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N samples (0 = no limit, for smoke tests)")
    parser.add_argument("--batch_size", type=int, default=50, help="Number of concurrent requests")
    parser.add_argument("--rpm_limit", type=int, default=300, help="Maximum requests per minute")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Maximum number of output tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (not supported by deepseek-reasoner)")
    parser.add_argument(
        "--model_name", type=str, default="deepseek-reasoner",
        help="DeepSeek model name (default: deepseek-reasoner, i.e. R1)",
    )
    parser.add_argument(
        "--api_url", type=str, default="https://api.deepseek.com/v1",
        help="DeepSeek API base URL",
    )
    parser.add_argument("--api_key_env", type=str, default="DEEPSEEK_API_KEY", help="Environment variable holding the API key")
    parser.add_argument("--cache_dir", type=str, default=None, help="HuggingFace cache directory")
    parser.add_argument(
        "--include_uuids", type=str, default=None,
        help="Only process the UUIDs in this file (JSONL format, with a uuid field)",
    )
    parser.add_argument(
        "--exclude_uuids", type=str, default=None,
        help="Exclude the UUIDs in this file (JSONL format, with a uuid field)",
    )
    args = parser.parse_args()

    output_path = project_root / args.output
    metadata_path = project_root / args.metadata_input
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_domains = set(args.domains.split(","))
    target_stages = args.stages.split(",")

    # Get the API key
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"ERROR: please set the environment variable {args.api_key_env}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("DeepSeek Rewrite: self-reflective CoT -> step-by-step solution")
    print(f"{'='*60}")
    print(f"Model:       {args.model_name}")
    print(f"API URL:    {args.api_url}")
    print(f"Output:      {output_path}")
    print(f"Concurrency: {args.batch_size}")
    print(f"RPM limit:   {args.rpm_limit}")
    print(f"max_tokens: {args.max_tokens}")
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

    # Load the include/exclude UUID filters
    include_set = None
    exclude_set = None
    if args.include_uuids:
        include_path = project_root / args.include_uuids
        include_set = set()
        with open(include_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    include_set.add(json.loads(line).get("uuid", ""))
        print(f"  include_uuids: {len(include_set)} (from {include_path})")

    if args.exclude_uuids:
        exclude_path = project_root / args.exclude_uuids
        exclude_set = set()
        with open(exclude_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    exclude_set.add(json.loads(line).get("uuid", ""))
        print(f"  exclude_uuids: {len(exclude_set)} (from {exclude_path})")

    # Filter
    pending = []
    for item in all_items:
        uuid = item["uuid"]
        if uuid in completed_uuids:
            continue
        if include_set is not None and uuid not in include_set:
            continue
        if exclude_set is not None and uuid in exclude_set:
            continue
        pending.append(item)

    print(f"  Total samples: {len(all_items)}")
    print(f"  Pending:       {len(pending)}")
    print(f"  Skipped:       {len(all_items) - len(pending)}")

    if args.limit > 0 and len(pending) > args.limit:
        pending = pending[:args.limit]
        print(f"  Applying --limit={args.limit}, actually processing: {len(pending)}")

    if not pending:
        print("\n  All samples have been processed.")
        return

    # ---- 3. Test API connectivity ----
    print(f"\n{'─'*40}")
    print("3. Testing API connectivity...")
    import urllib.request
    try:
        req = urllib.request.Request(
            f"{args.api_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"  API connection OK (status={resp.status})")
    except Exception as e:
        print(f"  WARNING: API connectivity test failed: {e}")
        print(f"  Will keep trying...")

    # ---- 4. Start rewriting ----
    print(f"\n{'─'*40}")
    print("4. Starting batch rewrite...")
    asyncio.run(
        rewrite_batch(
            items=pending,
            api_url=args.api_url,
            api_key=api_key,
            model_name=args.model_name,
            max_tokens=args.max_tokens,
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
