#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用 DeepSeek API 将 self-reflective CoT 答案改写为简洁 step-by-step 解答

将 Superior Reasoning 数据集中已有的长篇 self-reflective 推理过程，
喂给 DeepSeek V3.2，浓缩为 straight-forward 的 step-by-step 解题过程。

优势：利用已有的正确推理路径进行改写，而非从头生成，预期通过率远高于旧方案。

输入：
  - HuggingFace: Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b (stage1)
  - metadata.jsonl (uuid → ground_truth 映射)

输出：
  - no_think_deepseek_rewrite.jsonl（与 filter_no_think_data.py 兼容的格式）

使用方法:
    conda activate gpt-oss

    # 冒烟测试（20 条）
    python -m src.datasets.rewrite_no_think_deepseek --limit 20

    # 完整运行
    python -m src.datasets.rewrite_no_think_deepseek

依赖:
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
    print("请安装 aiohttp: pip install aiohttp")
    sys.exit(1)

# 项目根目录
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


# ==================== 反思性词汇检测 ====================

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
# 以下 pattern 不作为过滤条件（正常数学语言，非 self-reflective）:
# r'\bstep \d+\b'  — 结构化步骤标记
# r'\bconsider\b'  — 数学中的正常用语 (e.g., "consider the function f(x)")
# r'\bwe can\b', r'\bwe need to\b' — 正常解题语言
# r'\bnote that\b', r'\brecall that\b' — 引用定理/性质
# r'\bfirst\b.*\bthen\b' — 步骤顺序描述
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
    """从 HuggingFace 加载数据集，并关联 metadata 中的 ground_truth"""
    from datasets import load_dataset

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

    filtered = [item for item in all_data if item.get("domain", "") in domains]
    print(f"  过滤后 ({', '.join(domains)}): {len(filtered)} 条")

    gt_map = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    meta = json.loads(line)
                    gt_map[meta.get("uuid", "")] = meta.get("ground_truth")
        print(f"  metadata ground_truth: {len(gt_map)} 条")

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


# ==================== DeepSeek API 调用 ====================

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
    """调用 DeepSeek OpenAI-compatible API 改写单条样本

    Returns:
        (改写结果, 错误信息)
        成功: (text, None)
        失败: (None, {"error_type": ..., "error_message": ...})
    """
    prompt = REWRITE_PROMPT_TEMPLATE.format(question=question, solution=solution)

    url = f"{api_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # deepseek-reasoner 不支持 temperature/top_p 等采样参数
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
    # 仅非 reasoner 模式支持 temperature
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
                        # deepseek-reasoner 返回 reasoning_content (CoT) + content (最终答案)
                        # 我们只取 content（最终的 clean 答案）
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
                                "error_message": f"API 返回空内容, finish_reason={finish_reason}",
                                "details": {"attempts": attempt + 1},
                            }
                            if attempt < 2:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return (None, last_error)

                    elif resp.status == 429:
                        # 速率限制
                        wait_time = 30 * (attempt + 1)
                        print(f"  速率限制 (429)，等待 {wait_time}s...")
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
                    "error_message": "请求超时 (300s)",
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


# ==================== 批量处理 ====================

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
    """异步批量改写"""
    semaphore = asyncio.Semaphore(batch_size)
    rpm_delay = 60.0 / rpm_limit if rpm_limit > 0 else 0

    completed = 0
    failed = 0
    style_flagged = 0
    per_item_times = []
    start_time = time.time()

    # 小规模时每条都打印进度，大规模时每 100 条
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
                    f"成功:{completed} 失败:{failed} 风格异常:{style_flagged} | "
                    f"本条:{item_elapsed:.1f}s | "
                    f"平均:{sum(per_item_times)/len(per_item_times):.1f}s/条 | "
                    f"速度:{speed:.2f}条/秒 | "
                    f"剩余:{remaining/60:.1f}分钟"
                )

        tasks = [process_item(item) for item in items]
        await asyncio.gather(*tasks)

    f_out.close()

    elapsed = time.time() - start_time
    avg_time = sum(per_item_times) / len(per_item_times) if per_item_times else 0
    print(f"\n  {'='*50}")
    print(f"  改写完成!")
    print(f"  {'='*50}")
    print(f"  成功: {completed} | 失败: {failed} | 风格异常: {style_flagged}")
    print(f"  风格通过率: {(completed - style_flagged) / max(completed, 1) * 100:.1f}%")
    print(f"  总耗时: {elapsed:.1f}s ({elapsed / 60:.1f} 分钟)")
    print(f"  平均每条: {avg_time:.1f}s")
    print(f"  吞吐量: {(completed + failed) / max(elapsed, 0.01):.2f} 条/秒")
    print(f"  {'─'*50}")
    print(f"  65K 全量预估:")
    print(f"    时间: {65150 * avg_time / 3600:.1f} 小时 (串行)")
    print(f"    时间: {65150 / max((completed + failed) / max(elapsed, 0.01), 0.01) / 3600:.1f} 小时 (当前并发)")
    print(f"  {'='*50}")


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(
        description="使用 DeepSeek API 将 self-reflective CoT 答案改写为简洁 step-by-step 解答"
    )
    parser.add_argument(
        "--output", type=str,
        default="datasets/hybrid_superior_reasoning/no_think_deepseek_rewrite.jsonl",
        help="输出文件路径",
    )
    parser.add_argument(
        "--metadata_input", type=str,
        default="datasets/superior_reasoning/metadata.jsonl",
        help="metadata.jsonl 路径（用于 ground_truth，仍从 superior_reasoning 读取）",
    )
    parser.add_argument(
        "--dataset_name", type=str,
        default="Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b",
        help="HuggingFace 数据集名称",
    )
    parser.add_argument("--stages", type=str, default="stage1", help="数据集 stages")
    parser.add_argument("--domains", type=str, default="math,science", help="过滤 domains")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 条（0=不限制，用于冒烟测试）")
    parser.add_argument("--batch_size", type=int, default=50, help="并发请求数")
    parser.add_argument("--rpm_limit", type=int, default=300, help="每分钟最大请求数")
    parser.add_argument("--max_tokens", type=int, default=8192, help="最大输出 token 数")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度 (deepseek-reasoner 不支持此参数)")
    parser.add_argument(
        "--model_name", type=str, default="deepseek-reasoner",
        help="DeepSeek 模型名称 (默认: deepseek-reasoner, 即 R1)",
    )
    parser.add_argument(
        "--api_url", type=str, default="https://api.deepseek.com/v1",
        help="DeepSeek API base URL",
    )
    parser.add_argument("--api_key_env", type=str, default="DEEPSEEK_API_KEY", help="API key 环境变量名")
    parser.add_argument("--cache_dir", type=str, default=None, help="HuggingFace 缓存目录")
    parser.add_argument(
        "--include_uuids", type=str, default=None,
        help="仅处理此文件中的 UUID（JSONL 格式，含 uuid 字段）",
    )
    parser.add_argument(
        "--exclude_uuids", type=str, default=None,
        help="排除此文件中的 UUID（JSONL 格式，含 uuid 字段）",
    )
    args = parser.parse_args()

    output_path = project_root / args.output
    metadata_path = project_root / args.metadata_input
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_domains = set(args.domains.split(","))
    target_stages = args.stages.split(",")

    # 获取 API key
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"ERROR: 请设置环境变量 {args.api_key_env}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("DeepSeek Rewrite: 改写 self-reflective CoT → step-by-step 解答")
    print(f"{'='*60}")
    print(f"模型:       {args.model_name}")
    print(f"API URL:    {args.api_url}")
    print(f"输出:       {output_path}")
    print(f"并发数:     {args.batch_size}")
    print(f"RPM 限制:   {args.rpm_limit}")
    print(f"max_tokens: {args.max_tokens}")
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

    # 加载 include/exclude UUID 过滤
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
        print(f"  include_uuids: {len(include_set)} 条 (从 {include_path})")

    if args.exclude_uuids:
        exclude_path = project_root / args.exclude_uuids
        exclude_set = set()
        with open(exclude_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    exclude_set.add(json.loads(line).get("uuid", ""))
        print(f"  exclude_uuids: {len(exclude_set)} 条 (从 {exclude_path})")

    # 过滤
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

    print(f"  总样本数: {len(all_items)}")
    print(f"  待处理:   {len(pending)}")
    print(f"  已跳过:   {len(all_items) - len(pending)}")

    if args.limit > 0 and len(pending) > args.limit:
        pending = pending[:args.limit]
        print(f"  应用 --limit={args.limit}，实际处理: {len(pending)} 条")

    if not pending:
        print("\n  所有样本已处理完毕。")
        return

    # ---- 3. 测试 API 连通性 ----
    print(f"\n{'─'*40}")
    print("3. 测试 API 连通性...")
    import urllib.request
    try:
        req = urllib.request.Request(
            f"{args.api_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"  API 连通正常 (status={resp.status})")
    except Exception as e:
        print(f"  WARNING: API 连通测试失败: {e}")
        print(f"  将继续尝试...")

    # ---- 4. 开始改写 ----
    print(f"\n{'─'*40}")
    print("4. 开始批量改写...")
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
