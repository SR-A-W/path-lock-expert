#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用 Gemini API 将 self-reflective CoT 答案改写为简洁 step-by-step 解答

将 Superior Reasoning 数据集中已有的长篇 self-reflective 推理过程，
喂给 Gemini 3.1 Pro，浓缩为 straight-forward 的 step-by-step 解题过程。

优势：利用已有的正确推理路径进行改写，而非从头生成，预期通过率远高于旧方案。

输入：
  - HuggingFace: Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b (stage1)
  - metadata.jsonl (uuid → ground_truth 映射)

输出：
  - no_think_gemini_rewrite.jsonl（与 filter_no_think_data.py 兼容的格式）

使用方法:
    conda activate gpt-oss

    # 冒烟测试（20 条）
    python -m src.datasets.rewrite_no_think_gemini --limit 20

    # 完整运行
    python -m src.datasets.rewrite_no_think_gemini

依赖:
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
    print("请安装 google-genai: pip install google-genai")
    sys.exit(1)

# 项目根目录
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


# ==================== 反思性词汇检测（复用 filter_no_think_data.py 的模式）====================

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
    """检测文本是否包含反思性词汇，True = 通过（无反思词汇）"""
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(text):
            return False
    return True


# ==================== 答案提取 ====================

def extract_boxed_answer(text: str) -> str | None:
    """从文本中提取 \\boxed{...} 内的答案（栈匹配，支持嵌套大括号）"""
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


# ==================== 断点续传 ====================

def load_completed_uuids(output_path: str) -> set:
    """扫描已有输出文件，收集已完成的 uuid 集合"""
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


# ==================== 数据加载 ====================

def load_dataset_with_ground_truth(
    dataset_name: str,
    stages: list[str],
    domains: set[str],
    metadata_path: str,
    cache_dir: str | None = None,
) -> list[dict]:
    """从 HuggingFace 加载数据集，并关联 metadata 中的 ground_truth

    Returns:
        样本列表，每个样本包含 uuid, input, output, domain, ground_truth
    """
    from datasets import load_dataset

    # 加载 HuggingFace 数据集
    all_data = []
    for stage in stages:
        print(f"  加载 {stage}...")
        ds = load_dataset(dataset_name, stage, cache_dir=cache_dir)
        if isinstance(ds, dict):
            stage_data = ds.get("train", ds[list(ds.keys())[0]])
        else:
            stage_data = ds
        print(f"    {stage} 样本数: {len(stage_data)}")
        all_data.extend(stage_data)

    # 过滤 domain
    filtered = [item for item in all_data if item.get("domain", "") in domains]
    print(f"  过滤后 ({', '.join(domains)}): {len(filtered)} 条")

    # 加载 ground_truth 映射
    gt_map = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    meta = json.loads(line)
                    gt_map[meta.get("uuid", "")] = meta.get("ground_truth")
        print(f"  metadata ground_truth: {len(gt_map)} 条")

    # 合并
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


# ==================== Gemini API 调用 ====================

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
    """调用 Gemini API 改写单条样本

    Returns:
        (改写结果, 错误信息)
        成功: (text, None)
        失败: (None, {"error_type": ..., "error_message": ...})
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

                # 提取文本和完成状态
                text = None
                finish_reason = None
                try:
                    text = response.text
                except Exception:
                    # response.text 可能在 blocked 时抛异常
                    pass

                # 获取 finish_reason
                if response.candidates:
                    finish_reason = response.candidates[0].finish_reason

                if text:
                    # RPM 限流
                    await asyncio.sleep(rpm_delay)
                    # 检查是否截断
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
                        "error_message": f"Gemini 返回空文本, finish_reason={finish_reason}",
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
                # 速率限制错误需要更长等待
                if "ResourceExhausted" in error_name or "429" in str(e):
                    wait_time = 30 * (attempt + 1)
                    print(f"  速率限制，等待 {wait_time}s...")
                    await asyncio.sleep(wait_time)
                elif attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return (None, last_error)

    return (None, last_error)


# ==================== 批量处理 ====================

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
    """异步批量改写"""
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
            # 提取答案
            extracted_answer = extract_boxed_answer(result)

            # 内联 style check
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
            # 记录失败样本
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

        # 进度日志
        total_done = completed + failed
        if total_done % 100 == 0 and total_done > 0:
            elapsed = time.time() - start_time
            speed = total_done / elapsed
            remaining = (len(items) - total_done) / max(speed, 0.01)
            print(
                f"  进度: {total_done}/{len(items)} "
                f"({total_done/len(items)*100:.1f}%) | "
                f"成功: {completed} | 失败: {failed} | "
                f"风格异常: {style_flagged} | "
                f"速度: {speed:.1f} 条/秒 | "
                f"预计剩余: {remaining/60:.1f} 分钟"
            )

    # 并发执行
    tasks = [process_item(item) for item in items]
    await asyncio.gather(*tasks)

    f_out.close()

    # 最终统计
    elapsed = time.time() - start_time
    print(f"\n  改写完成!")
    print(f"  成功: {completed} | 失败: {failed} | 风格异常: {style_flagged}")
    print(f"  风格通过率: {(completed - style_flagged) / max(completed, 1) * 100:.1f}%")
    print(f"  总耗时: {elapsed / 60:.1f} 分钟")
    print(f"  平均速度: {(completed + failed) / max(elapsed, 0.01):.1f} 条/秒")


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(
        description="使用 Gemini API 将 self-reflective CoT 答案改写为简洁 step-by-step 解答"
    )
    parser.add_argument(
        "--output", type=str,
        default="datasets/superior_reasoning/no_think_gemini_rewrite.jsonl",
        help="输出文件路径",
    )
    parser.add_argument(
        "--metadata_input", type=str,
        default="datasets/superior_reasoning/metadata.jsonl",
        help="metadata.jsonl 路径（用于 ground_truth）",
    )
    parser.add_argument(
        "--dataset_name", type=str,
        default="Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b",
        help="HuggingFace 数据集名称",
    )
    parser.add_argument("--stages", type=str, default="stage1", help="数据集 stages")
    parser.add_argument("--domains", type=str, default="math,science", help="过滤 domains")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 条（0=不限制，用于冒烟测试）")
    parser.add_argument("--batch_size", type=int, default=8, help="并发请求数")
    parser.add_argument("--rpm_limit", type=int, default=60, help="每分钟最大请求数")
    parser.add_argument("--max_output_tokens", type=int, default=4096, help="最大输出 token 数")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度")
    parser.add_argument("--model_name", type=str, default="gemini-3.1-pro-preview", help="Gemini 模型名称")
    parser.add_argument("--api_key_env", type=str, default="GEMINI_API_KEY", help="API key 环境变量名")
    parser.add_argument("--cache_dir", type=str, default=None, help="HuggingFace 缓存目录")
    args = parser.parse_args()

    # 解析路径
    output_path = project_root / args.output
    metadata_path = project_root / args.metadata_input
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_domains = set(args.domains.split(","))
    target_stages = args.stages.split(",")

    # 获取 API key
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"ERROR: 请设置环境变量 {args.api_key_env}")
        print(f"  export {args.api_key_env}='your-api-key'")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Gemini Rewrite: 改写 self-reflective CoT → step-by-step 解答")
    print(f"{'='*60}")
    print(f"模型:       {args.model_name}")
    print(f"输出:       {output_path}")
    print(f"并发数:     {args.batch_size}")
    print(f"RPM 限制:   {args.rpm_limit}")
    print(f"max_tokens: {args.max_output_tokens}")
    print(f"temperature: {args.temperature}")
    if args.limit > 0:
        print(f"限制数量:   仅处理前 {args.limit} 条")

    # ---- 1. 断点续传检查 ----
    print(f"\n{'─'*40}")
    print("1. 检查断点续传状态...")
    completed_uuids = load_completed_uuids(str(output_path))
    if completed_uuids:
        print(f"  发现 {len(completed_uuids)} 条已完成记录，将跳过")
    else:
        print(f"  无已完成记录，从头开始")

    # ---- 2. 加载数据集 ----
    print(f"\n{'─'*40}")
    print("2. 加载数据集...")
    all_items = load_dataset_with_ground_truth(
        dataset_name=args.dataset_name,
        stages=target_stages,
        domains=target_domains,
        metadata_path=str(metadata_path),
        cache_dir=args.cache_dir,
    )

    # 过滤已完成
    pending = [item for item in all_items if item["uuid"] not in completed_uuids]
    print(f"  总样本数: {len(all_items)}")
    print(f"  待处理:   {len(pending)}")
    print(f"  已跳过:   {len(all_items) - len(pending)}")

    # 应用 --limit
    if args.limit > 0 and len(pending) > args.limit:
        pending = pending[:args.limit]
        print(f"  应用 --limit={args.limit}，实际处理: {len(pending)} 条")

    if not pending:
        print("\n  所有样本已处理完毕。")
        return

    # ---- 3. 初始化 Gemini client ----
    print(f"\n{'─'*40}")
    print("3. 初始化 Gemini client...")
    client = genai.Client(api_key=api_key)

    # 测试连通性
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
        print(f"  API 连通正常: {test_text[:50]}")
    except Exception as e:
        print(f"  WARNING: API 连通测试失败: {e}")
        print(f"  将继续尝试...")

    # ---- 4. 开始改写 ----
    print(f"\n{'─'*40}")
    print("4. 开始批量改写...")
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

    # ---- 5. 统计 ----
    print(f"\n{'─'*40}")
    print("5. 输出统计...")
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

    print(f"  输出文件总行数: {total_out}")
    print(f"  成功生成: {success_out}")
    print(f"  风格通过: {style_pass} ({style_pass / max(success_out, 1) * 100:.1f}%)")
    print(f"  输出文件: {output_path}")

    print(f"\n{'='*60}")
    print("改写完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
