#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-generate /no_think answers through a vLLM API

Reads metadata.jsonl (question + ground truth) produced by prepare_think_data.py,
and calls a vLLM-served model (e.g. Qwen3-32B-Instruct) to generate short, direct answers.

Only for the math + science domains (instruction-following is extracted directly in Step 1).

Supports asynchronous concurrent requests, resuming from a checkpoint, and topping up rejected samples.

Usage:

    # First generation
    python -m src.datasets.generate_no_think_data \
        --metadata_input datasets/superior_reasoning/metadata.jsonl \
        --output datasets/superior_reasoning/no_think_raw.jsonl \
        --api_url http://localhost:8000/v1 \
        --model_name Qwen3-32B

    # Top up rejected samples (regenerate them)
    python -m src.datasets.generate_no_think_data \
        --retry_from datasets/superior_reasoning/rejected_uuids.jsonl

    # Smoke test (generate only 50 samples)
    python -m src.datasets.generate_no_think_data \
        --limit 50 \
        --output datasets/superior_reasoning/no_think_smoke_test.jsonl

Dependencies:
    pip install aiohttp
"""

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
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


# ==================== Constants ====================

# System prompt: force the model to answer directly, without a reasoning process
SYSTEM_PROMPT = """You are a helpful assistant that help with answering questions in a concise and direct way."""
MATH_INPUT_PROMPT = """
Answer the question in brief, concise, and direct ways. You should follow the style requirements:
- State each reasoning step only once: do NOT repeat or rephrase any step you have already written.
- Do NOT revise, second-guess, or reflect on your own reasoning mid-response. Do NOT use hesitation markers such as "Wait", "Hmm", "Actually", "Alternatively" and so on).
- Once you have reached your final conclusion, put the final answer in \\\\boxed{}."""
SCIENCE_INPUT_PROMPT = """
Answer the question in brief, concise, and direct ways. You should follow the style requirements:
- State each reasoning step only once: do NOT repeat or rephrase any step you have already written.
- Do NOT revise, second-guess, or reflect on your own reasoning mid-response. Do NOT use hesitation markers such as "Wait", "Hmm", "Actually", "Alternatively" and so on).
- Once you have reached your final conclusion, make sure to put the answer inside \\\\boxed{}."""


# ==================== Resume from checkpoint ====================

def load_completed_uuids(output_path: str) -> set:
    """Scan the existing output file and collect the set of completed uuids (for resuming)

    Args:
        output_path: output file path

    Returns:
        set of completed uuids
    """
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


def load_metadata(metadata_path: str, completed_uuids: set) -> tuple[list, int]:
    """Load the metadata, skipping completed samples

    Args:
        metadata_path: path to metadata.jsonl
        completed_uuids: set of completed uuids

    Returns:
        (list of pending samples, total number of samples)
    """
    pending = []
    total = 0
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            total += 1
            if item.get("uuid") not in completed_uuids:
                pending.append(item)

    return pending, total


# ==================== Top-up mechanism ====================

def load_retry_uuids(retry_path: str) -> set:
    """Load the set of UUIDs to regenerate from rejected_uuids.jsonl

    Args:
        retry_path: path to rejected_uuids.jsonl

    Returns:
        set of UUIDs to regenerate
    """
    uuids = set()
    with open(retry_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "uuid" in item:
                uuids.add(item["uuid"])
    return uuids


def remove_uuids_from_output(output_path: str, uuids_to_remove: set) -> int:
    """Remove the old records of the given UUIDs from the output file

    Uses atomic replacement via a temporary file to avoid corrupting the data.

    Args:
        output_path: output file path
        uuids_to_remove: set of UUIDs to remove

    Returns:
        number of records removed
    """
    if not os.path.exists(output_path):
        return 0

    removed = 0
    # Write a temporary file, then replace atomically
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(output_path), suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f_tmp, \
             open(output_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                try:
                    item = json.loads(line_stripped)
                    if item.get("uuid") in uuids_to_remove:
                        removed += 1
                        continue
                except json.JSONDecodeError:
                    pass
                f_tmp.write(line_stripped + "\n")
        os.replace(tmp_path, output_path)
    except Exception:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return removed


def load_retry_items(metadata_path: str, retry_uuids: set) -> list:
    """Find the samples to regenerate in metadata.jsonl

    Args:
        metadata_path: path to metadata.jsonl
        retry_uuids: set of UUIDs to regenerate

    Returns:
        list of samples to regenerate
    """
    items = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("uuid") in retry_uuids:
                items.append(item)
    return items


# ==================== Input preprocessing ====================

def preprocess_input(input_text: str, domain: str) -> str:
    """Preprocess the input text, replacing specific instruction phrases

    Args:
        input_text: the original input text
        domain: data domain (math or science)

    Returns:
        the processed input text
    """
    if domain == "math":
        # Math data: replace the instruction at the end
        pattern = r"Let's think step by step and output the final answer within \\boxed\{\}\.\s*$"
        if re.search(pattern, input_text):
            input_text = re.sub(pattern, MATH_INPUT_PROMPT, input_text)
    elif domain == "science":
        # Science data: replace the instruction at the beginning
        pattern = r"^\s*Solve the following problem\. Make sure to put the answer \(and only answer\) inside \\boxed\{\}\.\s*"
        if re.search(pattern, input_text):
            input_text = re.sub(pattern, SCIENCE_INPUT_PROMPT + " ", input_text)

    return input_text


# ==================== API calls ====================

def extract_boxed_answer(text: str) -> str | None:
    """Extract the answer inside \\boxed{...} from text (stack matching, nested braces supported)

    Args:
        text: text containing \\boxed{}

    Returns:
        the extracted answer string, or None if not found
    """
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


def extract_answer_heuristic(text: str, domain: str) -> str | None:
    """Heuristically extract the answer (when \\boxed{} is absent)

    Args:
        text: text generated by the model
        domain: data domain

    Returns:
        the extracted answer, or None if not found
    """
    # science: try to match a multiple-choice letter
    if domain == "science":
        m = re.search(r'\b([A-E])\b\s*$', text.strip())
        if m:
            return m.group(1).upper()

    # Try the "answer is X" pattern
    for pat in [
        r'(?:the\s+)?answer\s+is\s*[:\s]*(.+?)(?:\.|$)',
        r'(?:therefore|thus|hence)\s*,?\s*(.+?)(?:\.|$)',
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # If the text is very short, the whole text may be the answer
    stripped = text.strip()
    if len(stripped) <= 50:
        return stripped

    return None


def extract_generated_answer(text: str, domain: str) -> str | None:
    """Extract the answer from generated text

    Args:
        text: text generated by the model
        domain: data domain

    Returns:
        the extracted answer, or None if not found
    """
    answer = extract_boxed_answer(text)
    if answer is not None:
        return answer
    return extract_answer_heuristic(text, domain)


async def call_chat_api(
    session: aiohttp.ClientSession,
    api_url: str,
    model_name: str,
    user_message: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    semaphore: asyncio.Semaphore,
) -> tuple[str | None, dict | None]:
    """Call the vLLM OpenAI-compatible chat API (a single request)

    Args:
        session: aiohttp session
        api_url: API base URL (e.g. http://localhost:8000/v1)
        model_name: model name
        user_message: user message
        system_prompt: system prompt
        max_tokens: maximum number of generated tokens
        temperature: sampling temperature
        timeout: request timeout in seconds
        semaphore: semaphore for concurrency control

    Returns:
        (generated text, error info dict)
        success: (text, None)
        failure: (None, {"error_type": ..., "error_message": ..., "details": ...})
    """
    url = f"{api_url}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        # Disable Qwen3's built-in thinking mode
        "chat_template_kwargs": {"enable_thinking": False},
    }

    last_error = None
    async with semaphore:
        for attempt in range(3):  # retry at most 3 times
            try:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"]

                        # Check whether content is None
                        if content is None:
                            last_error = {
                                "error_type": "api_error",
                                "error_message": "API returned content as None",
                                "details": {
                                    "finish_reason": data["choices"][0].get("finish_reason", ""),
                                    "attempts": attempt + 1,
                                }
                            }
                            if attempt < 2:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return (None, last_error)

                        # Check whether the output was truncated
                        finish_reason = data["choices"][0].get("finish_reason", "")
                        usage = data.get("usage", {})
                        completion_tokens = usage.get("completion_tokens", 0)

                        if finish_reason == "length":
                            # The output was truncated
                            return (content, {
                                "error_type": "truncated",
                                "error_message": "Output truncated (max_tokens limit reached)",
                                "details": {
                                    "finish_reason": finish_reason,
                                    "max_tokens": max_tokens,
                                    "completion_tokens": completion_tokens,
                                    "output_length_chars": len(content),
                                }
                            })

                        return (content, None)
                    else:
                        error_text = await resp.text()
                        last_error = {
                            "error_type": "api_error",
                            "error_message": f"API returned error status code {resp.status}",
                            "details": {
                                "status_code": resp.status,
                                "response_text": error_text[:500],
                                "attempts": attempt + 1,
                            }
                        }
                        if attempt < 2:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        print(f"  API error (status={resp.status}): {error_text[:200]}")
                        return (None, last_error)
            except asyncio.TimeoutError as e:
                last_error = {
                    "error_type": "timeout",
                    "error_message": f"Request timed out ({timeout}s)",
                    "details": {
                        "timeout_seconds": timeout,
                        "attempts": attempt + 1,
                    }
                }
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                print(f"  Request timed out: {e}")
                return (None, last_error)
            except aiohttp.ClientError as e:
                last_error = {
                    "error_type": "network_error",
                    "error_message": f"Network error: {type(e).__name__}",
                    "details": {
                        "exception": str(e),
                        "attempts": attempt + 1,
                    }
                }
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                print(f"  Network exception: {e}")
                return (None, last_error)
            except Exception as e:
                last_error = {
                    "error_type": "unknown_error",
                    "error_message": f"Unknown error: {type(e).__name__}",
                    "details": {
                        "exception": str(e),
                        "attempts": attempt + 1,
                    }
                }
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                print(f"  Unknown exception: {e}")
                return (None, last_error)

    return (None, last_error)


# ==================== Batch generation ====================

async def generate_batch(
    items: list,
    api_url: str,
    model_name: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    batch_size: int,
    output_path: str,
    failed_output_path: str,
    rejected_output_path: str,
    is_retry_mode: bool = False,
    retry_uuids: set = None,
):
    """Generate /no_think answers in asynchronous batches

    Args:
        items: list of samples to process
        api_url: vLLM API URL
        model_name: model name
        system_prompt: system prompt
        max_tokens: maximum number of generated tokens
        temperature: sampling temperature
        timeout: request timeout in seconds
        batch_size: number of concurrent requests
        output_path: output file path
        failed_output_path: output path for failed-sample details (JSON format)
        rejected_output_path: output path for rejected UUIDs (JSONL format, used by retry)
        is_retry_mode: whether this is retry mode
        retry_uuids: UUIDs to regenerate in retry mode
    """
    semaphore = asyncio.Semaphore(batch_size)
    completed = 0
    failed = 0
    truncated = 0
    start_time = time.time()

    # Open the output file in append mode (supports resuming)
    f_out = open(output_path, "a", encoding="utf-8")

    # Collect failed samples
    failed_samples = []

    async with aiohttp.ClientSession() as session:

        async def process_item(item):
            nonlocal completed, failed, truncated
            uuid = item.get("uuid", "")
            input_text = item.get("input", "")
            domain = item.get("domain", "")
            ground_truth = item.get("ground_truth")

            # Preprocess the input text (replace specific instruction phrases)
            processed_input = preprocess_input(input_text, domain)

            result, error_info = await call_chat_api(
                session, api_url, model_name, processed_input,
                system_prompt, max_tokens, temperature, timeout, semaphore,
            )

            if result is not None:
                # Extract the answer from the generated output
                extracted_answer = extract_generated_answer(result, domain)

                output_item = {
                    "uuid": uuid,
                    "input": processed_input,  # save the preprocessed input
                    "domain": domain,
                    "generated_output": result,
                    "extracted_answer": extracted_answer,  # extracted answer
                    "ground_truth": ground_truth,
                }
                f_out.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                f_out.flush()
                completed += 1

                # Check whether the output was truncated
                if error_info and error_info.get("error_type") == "truncated":
                    truncated += 1
                    failed_samples.append({
                        "uuid": uuid,
                        "input": processed_input,  # save the preprocessed input
                        "domain": domain,
                        "ground_truth": ground_truth,
                        "generated_output": result,
                        "extracted_answer": extracted_answer,
                        **error_info,
                    })
            else:
                failed += 1
                # Record failed samples
                failed_samples.append({
                    "uuid": uuid,
                    "input": processed_input,  # save the preprocessed input
                    "domain": domain,
                    "ground_truth": ground_truth,
                    **(error_info or {"error_type": "unknown", "error_message": "Unknown error"}),
                })

            # Progress log (every 1000 items)
            total_done = completed + failed
            if total_done % 1000 == 0 and total_done > 0:
                elapsed = time.time() - start_time
                speed = total_done / elapsed
                remaining = (len(items) - total_done) / max(speed, 0.01)
                print(
                    f"  Progress: {total_done}/{len(items)} "
                    f"({total_done/len(items)*100:.1f}%) | "
                    f"Succeeded: {completed} | Failed: {failed} | Truncated: {truncated} | "
                    f"Speed: {speed:.1f} items/s | "
                    f"ETA: {remaining/60:.1f} min"
                )

        # Create all tasks and run them concurrently
        tasks = [process_item(item) for item in items]
        await asyncio.gather(*tasks)

    f_out.close()

    # Handle the failed-sample records (retry mode needs to merge the old records)
    if is_retry_mode and retry_uuids:
        # Retry mode: load the old failure records, remove the retried UUIDs, then merge the new failure records
        old_failed_samples = []
        if os.path.exists(failed_output_path):
            with open(failed_output_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                old_failed_samples = old_data.get("failed_samples", [])

        # Remove the UUIDs being retried
        old_failed_samples = [s for s in old_failed_samples if s.get("uuid") not in retry_uuids]

        # Merge the new failure records
        all_failed_samples = old_failed_samples + failed_samples

        # Handle rejected_uuids.jsonl the same way
        old_rejected_items = []
        if os.path.exists(rejected_output_path):
            with open(rejected_output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        if item.get("uuid") not in retry_uuids:
                            old_rejected_items.append(item)
    else:
        # Normal mode: use the new failure records directly
        all_failed_samples = failed_samples
        old_rejected_items = []

    # Save the failed samples to a JSON file (detailed info)
    if all_failed_samples:
        # Count error types
        error_type_counts = {}
        for sample in all_failed_samples:
            error_type = sample.get("error_type", "unknown")
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1

        truncated_count = sum(1 for s in all_failed_samples if s.get("error_type") == "truncated")

        with open(failed_output_path, "w", encoding="utf-8") as f_failed:
            json.dump({
                "summary": {
                    "total_failed": len(all_failed_samples),
                    "truncated_count": truncated_count,
                    "error_types": error_type_counts,
                },
                "failed_samples": all_failed_samples,
            }, f_failed, ensure_ascii=False, indent=2)

        # Save rejected_uuids.jsonl (used by retry)
        with open(rejected_output_path, "w", encoding="utf-8") as f_rejected:
            # Write the old records first
            for item in old_rejected_items:
                f_rejected.write(json.dumps(item, ensure_ascii=False) + "\n")
            # Then write the new failure records
            for sample in failed_samples:
                rejected_item = {
                    "uuid": sample.get("uuid", ""),
                    "domain": sample.get("domain", "unknown"),
                    "reason": f"generation_failed_{sample.get('error_type', 'unknown')}",
                    "input": sample.get("input", ""),
                }
                f_rejected.write(json.dumps(rejected_item, ensure_ascii=False) + "\n")

    # Final statistics
    elapsed = time.time() - start_time
    print(f"\n  Generation finished!")
    print(f"  Succeeded: {completed} | Failed: {failed} | Truncated: {truncated}")
    print(f"  Total time: {elapsed/60:.1f} min")
    print(f"  Average speed: {(completed + failed) / max(elapsed, 0.01):.1f} items/s")
    if all_failed_samples:
        print(f"  Failure details saved to: {failed_output_path}")
        print(f"  Rejected UUIDs saved to: {rejected_output_path}")
        if is_retry_mode:
            print(f"  (Retry mode: failure records updated, successful UUIDs removed)")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Batch-generate /no_think answers through a vLLM API (math + science only)"
    )
    parser.add_argument(
        "--metadata_input",
        type=str,
        default="datasets/superior_reasoning/metadata.jsonl",
        help="Metadata file path (default: datasets/superior_reasoning/metadata.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/superior_reasoning/no_think_raw.jsonl",
        help="Output file path (default: datasets/superior_reasoning/no_think_raw.jsonl)",
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen3-32B",
        help="Model name (default: Qwen3-32B)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Number of concurrent requests (default: 256)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of generated tokens (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0, greedy decoding)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout of a single request in seconds (default: 300)",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Custom system prompt (default: the built-in prompt)",
    )
    parser.add_argument(
        "--retry_from",
        type=str,
        default=None,
        help="Path to rejected_uuids.jsonl; only the rejected samples are regenerated",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N samples (0 = no limit, for smoke tests)",
    )
    parser.add_argument(
        "--failed_output",
        type=str,
        default=None,
        help="Output path for failed-sample details (JSON format, generated automatically by default)",
    )
    parser.add_argument(
        "--rejected_output",
        type=str,
        default=None,
        help="Output path for rejected UUIDs (JSONL format, generated automatically by default, used by retry)",
    )
    args = parser.parse_args()

    # Resolve paths
    metadata_path = project_root / args.metadata_input
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Output path for failed-sample details (default: same directory as the output file, _failed.json suffix)
    if args.failed_output:
        failed_output_path = project_root / args.failed_output
    else:
        failed_output_path = output_path.parent / (output_path.stem + "_failed.json")

    # Output path for rejected UUIDs (default: rejected_uuids.jsonl)
    if args.rejected_output:
        rejected_output_path = project_root / args.rejected_output
    else:
        rejected_output_path = output_path.parent / "rejected_uuids.jsonl"

    system_prompt = args.system_prompt if args.system_prompt else SYSTEM_PROMPT

    print(f"\n{'='*60}")
    print("Step 2: Generate /no_think answers")
    print(f"{'='*60}")
    print(f"Metadata file:  {metadata_path}")
    print(f"Output file:    {output_path}")
    print(f"Failure details: {failed_output_path}")
    print(f"Rejected UUIDs: {rejected_output_path}")
    print(f"API URL:    {args.api_url}")
    print(f"Model:          {args.model_name}")
    print(f"Concurrency:    {args.batch_size}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"temperature: {args.temperature}")
    print(f"timeout:        {args.timeout}s")
    if args.retry_from:
        print(f"Top-up mode:    reading rejected UUIDs from {args.retry_from}")
    if args.limit > 0:
        print(f"Limit:          only the first {args.limit} samples")

    if args.retry_from:
        # ---- Top-up mode ----
        retry_path = project_root / args.retry_from
        print(f"\n{'─'*40}")
        print("Top-up mode: regenerating rejected samples")

        # 1. Read the rejected UUIDs
        retry_uuids = load_retry_uuids(str(retry_path))
        print(f"  Number of rejected UUIDs: {len(retry_uuids)}")

        if not retry_uuids:
            print("  Nothing to top up, exiting.")
            return

        # 2. Remove the old records from the output file
        removed = remove_uuids_from_output(str(output_path), retry_uuids)
        print(f"  Removed old records from the output file: {removed}")

        # 3. Find the matching samples in metadata
        pending = load_retry_items(str(metadata_path), retry_uuids)
        print(f"  Samples found to regenerate: {len(pending)}")

        if not pending:
            print("  No matching samples found in metadata, exiting.")
            return

        # Mark as retry mode
        is_retry_mode = True
    else:
        # ---- Normal mode ----
        # 1. Load the completed uuids (resume from checkpoint)
        print(f"\n{'─'*40}")
        print("1. Checking resume state...")
        completed_uuids = load_completed_uuids(str(output_path))
        if completed_uuids:
            print(f"  Found {len(completed_uuids)} completed records, these samples will be skipped")
        else:
            print("  No completed records, starting from scratch")

        # 2. Load the pending samples
        print(f"\n{'─'*40}")
        print("2. Loading pending samples...")
        pending, total = load_metadata(str(metadata_path), completed_uuids)
        print(f"  Total metadata entries: {total}")
        print(f"  Pending:       {len(pending)}")
        print(f"  Skipped:       {total - len(pending)}")

        # Mark as normal mode
        is_retry_mode = False
        retry_uuids = set()

    # Apply --limit
    if args.limit > 0 and len(pending) > args.limit:
        pending = pending[:args.limit]
        print(f"  Applying --limit={args.limit}, actually processing: {len(pending)}")

    if not pending:
        print("\n  All samples have been processed, nothing to generate.")
        return

    # ---- Test API connectivity ----
    print(f"\n{'─'*40}")
    print("Testing API connectivity...")
    import urllib.request
    try:
        req = urllib.request.Request(f"{args.api_url}/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"  API connection OK (status={resp.status})")
    except Exception as e:
        print(f"  Warning: API connectivity test failed: {e}")
        print(f"  Will keep trying to generate, but errors may occur...")

    # ---- Start batch generation ----
    print(f"\n{'─'*40}")
    print("Starting batch generation...")
    asyncio.run(
        generate_batch(
            items=pending,
            api_url=args.api_url,
            model_name=args.model_name,
            system_prompt=system_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            batch_size=args.batch_size,
            output_path=str(output_path),
            failed_output_path=str(failed_output_path),
            rejected_output_path=str(rejected_output_path),
            is_retry_mode=is_retry_mode,
            retry_uuids=retry_uuids,
        )
    )

    print(f"\n{'='*60}")
    print("Step 2 done!")
    print(f"Output file: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
