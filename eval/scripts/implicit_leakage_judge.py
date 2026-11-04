#!/usr/bin/env python3

import argparse
import asyncio
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp

LEAKAGE_JUDGE_PROMPT = """You are an expert evaluator measuring reasoning leakage in a hybrid-thinking language model. The model was instructed to answer in "no-thinking" mode (a direct answer without chain-of-thought deliberation). Heuristic keyword counting (e.g. "wait", "hmm") under-detects implicit or paraphrased reasoning, so you must judge the SEMANTIC content.

Classify the response below into exactly ONE of three mutually exclusive categories:

1. NO_REASONING: The model provides a direct answer without any intermediate deliberation (optionally a single brief justification line).
2. IMPLICIT_REASONING: The <think></think> block is empty or absent, BUT chain-of-thought reasoning, reflection, multi-step derivation, case analysis, or verification is semantically present in the standard output block. (This is reasoning leakage.)
3. EXPLICIT_REASONING: Step-by-step reasoning is properly contained entirely within <think></think> blocks.

**Model's response (no-thinking mode):**
{response}

Output ONLY your verdict in this exact format and nothing else: \\boxed{{NO_REASONING}} or \\boxed{{IMPLICIT_REASONING}} or \\boxed{{EXPLICIT_REASONING}}."""

_VALID_LABELS = {"NO_REASONING", "IMPLICIT_REASONING", "EXPLICIT_REASONING"}

def extract_response_text(record: dict) -> Optional[str]:
    try:
        mo = record.get("model_output") or {}
        choices = mo.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            c = msg.get("content")
            if c:
                return c
    except (AttributeError, IndexError, TypeError):
        pass
    try:
        msgs = record.get("messages") or []
        if msgs and isinstance(msgs[0], dict):
            c = msgs[0].get("content")
            if c:
                return c
    except (AttributeError, IndexError, TypeError):
        pass
    return None

def iter_prediction_files(input_dir: Path) -> List[Path]:
    return sorted(input_dir.rglob("predictions/**/*.jsonl"))

def parse_model_bench(pred_path: Path) -> Tuple[str, str, str]:
    parts = pred_path.parts
    try:
        ts_idx = parts.index("predictions")
        mode = parts[ts_idx - 2]
        bench = parts[ts_idx - 3]
        model = parts[ts_idx - 4]
        return model, bench, mode
    except (ValueError, IndexError):
        return "unknown_model", "unknown_bench", "unknown_mode"

def _parse_label(content: str) -> Optional[str]:
    m = re.search(r"\\boxed\s*\{([^}]+)\}", content)
    cand = m.group(1).strip().upper() if m else content.upper()
    cand = cand.replace("-", "_").replace(" ", "_")
    for lab in _VALID_LABELS:
        if lab in cand:
            return lab
    return None

async def judge_one(
    session: aiohttp.ClientSession,
    response_text: str,
    api_url: str,
    model_name: str,
    timeout: int = 90,
    max_chars: int = 12000,
) -> Optional[str]:
    prompt = LEAKAGE_JUDGE_PROMPT.format(response=response_text[:max_chars])
    url = f"{api_url}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
        "temperature": 0.0,
    }
    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            return _parse_label(data["choices"][0]["message"]["content"])
    except Exception:
        return None

async def judge_responses(
    responses: List[str], api_url: str, model_name: str, concurrency: int = 8
) -> List[Optional[str]]:
    sem = asyncio.Semaphore(concurrency)
    results: List[Optional[str]] = [None] * len(responses)
    async with aiohttp.ClientSession() as session:
        async def worker(i: int, text: str):
            async with sem:
                results[i] = await judge_one(session, text, api_url, model_name)
        await asyncio.gather(*(worker(i, t) for i, t in enumerate(responses)))
    return results

def write_back_report(run_dir: Path, stats: dict) -> int:
    written = 0
    for rep in run_dir.rglob("reports/**/*.json"):
        try:
            with open(rep) as f:
                d = json.load(f)
            d["semantic_leakage"] = stats
            with open(rep, "w") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
            written += 1
        except (json.JSONDecodeError, OSError):
            continue
    return written

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", nargs="+", required=True,
                    help="One or more directories; predictions/*.jsonl are searched recursively")
    ap.add_argument("--judge-api-url", default="http://localhost:8000/v1")
    ap.add_argument("--judge-model-name", default="gpt-oss-20b")
    ap.add_argument("--mode-filter", default="no_think",
                    help="Only judge predictions of this mode (default: no_think)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--no-writeback", action="store_true",
                    help="Do not write back to reports/*.json; only print and emit the aggregated JSON")
    args = ap.parse_args()

    bucket: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    run_dirs: Dict[Tuple[str, str], set] = defaultdict(set)

    for d in args.input_dir:
        base = Path(d)
        if not base.exists():
            print(f"[WARN] input dir missing: {base}")
            continue
        for pf in iter_prediction_files(base):
            model, bench, mode = parse_model_bench(pf)
            if args.mode_filter and mode != args.mode_filter:
                continue
            try:
                run_dir = pf.parents[list(pf.parts).index("predictions") - len(pf.parts)]
            except Exception:
                run_dir = pf.parent
            run_dir = pf.parent
            while run_dir.name != "predictions" and run_dir.parent != run_dir:
                run_dir = run_dir.parent
            run_dir = run_dir.parent
            with open(pf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    txt = extract_response_text(rec)
                    if txt:
                        bucket[(model, bench)].append(txt)
                        run_dirs[(model, bench)].add(run_dir)

    if not bucket:
        print("[ERROR] no predictions found for mode-filter="
              f"{args.mode_filter}. Check --input-dir / mode.")
        return

    summary = {}
    print(f"\n{'model':>45s} | {'bench':<14s} | impl_leak | (none/impl/expl/fail of N)")
    for (model, bench), texts in sorted(bucket.items()):
        verdicts = asyncio.run(
            judge_responses(texts, args.judge_api_url, args.judge_model_name, args.concurrency)
        )
        n_none = sum(1 for v in verdicts if v == "NO_REASONING")
        n_impl = sum(1 for v in verdicts if v == "IMPLICIT_REASONING")
        n_expl = sum(1 for v in verdicts if v == "EXPLICIT_REASONING")
        n_fail = sum(1 for v in verdicts if v is None)
        n = n_none + n_impl + n_expl
        impl_rate = (n_impl / n) if n else 0.0
        stats = {
            "model": model, "benchmark": bench, "mode": args.mode_filter,
            "implicit_leakage_rate": impl_rate,
            "n_judged": n, "n_no_reasoning": n_none,
            "n_implicit_reasoning": n_impl, "n_explicit_reasoning": n_expl,
            "n_judge_fail": n_fail,
            "rate_no_reasoning": (n_none / n) if n else 0.0,
            "rate_explicit_reasoning": (n_expl / n) if n else 0.0,
            "judge_model": args.judge_model_name,
            "method": "Demystifying-3class-LLM-judge",
        }
        summary[f"{model}/{bench}"] = stats
        print(f"{model:>45s} | {bench:<14s} | {impl_rate:8.3f}  "
              f"(none {n_none}/impl {n_impl}/expl {n_expl}/fail {n_fail} of {n})")
        if not args.no_writeback:
            for rd in run_dirs[(model, bench)]:
                write_back_report(rd, stats)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] aggregate written to {args.out_json}")

if __name__ == "__main__":
    main()
