#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通过 vLLM API 批量生成 /no_think 模式的回答

读取 prepare_think_data.py 产出的 metadata.jsonl（问题 + ground truth），
调用 vLLM 部署的模型（如 Qwen3-32B-Instruct）生成简短直接的回答。

仅针对 math + science domain（instruction-following 在 Step 1 中直接提取）。

支持异步并发请求、断点续传、被拒绝样本补足。

使用方法:
    conda activate het

    # 首次生成
    python -m src.datasets.generate_no_think_data \
        --metadata_input datasets/superior_reasoning/metadata.jsonl \
        --output datasets/superior_reasoning/no_think_raw.jsonl \
        --api_url http://localhost:8000/v1 \
        --model_name Qwen3-32B

    # 补足被拒绝的样本（重新生成）
    python -m src.datasets.generate_no_think_data \
        --retry_from datasets/superior_reasoning/rejected_uuids.jsonl

    # 冒烟测试（仅生成 50 条）
    python -m src.datasets.generate_no_think_data \
        --limit 50 \
        --output datasets/superior_reasoning/no_think_smoke_test.jsonl

依赖:
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
    print("请安装 aiohttp: pip install aiohttp")
    sys.exit(1)

# 项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ==================== 常量 ====================

# System prompt：强制模型直接回答，禁止推理过程
# NO_THINK_PROMPT = f"""Please provide direct, concise answers for the question and following the rules.

# Rules:
# 1. Give the final answer immediately. Do NOT show any long self-reflective Chain of Thought reasoning steps, working, or thought process.
# 2. Do NOT use phrases like "Let me think", "First", "Step 1", "We need to", "Let's", "Hmm", "Wait", "Okay so".
# 3. You can use very concise and straight-forward reasoning steps, but put the final answer inside \\boxed{}.
# 4. Keep your response short and concise.

# The question is: 
SYSTEM_PROMPT = """You are a helpful assistant that help with answering questions in a concise and direct way."""
# SYSTEM_PROMPT = """Answer the following question with brief, concise, and direct reasoning.

# Style requirements:
# - Write in a linear, forward-only style: state your reasoning steps once, then give the final answer.
# - Do NOT revise, second-guess, or reflect on your own reasoning mid-response. Write as if you are confident from the start.
# - Do NOT use hesitation markers such as "Wait", "Hmm", "Actually", "Alternatively", "Let me reconsider", or "On second thought".
# - Put the final answer in \\boxed{}.
# - Keep your response focused: present only the reasoning steps necessary to reach the answer, without padding or repetition.
# """
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


# ==================== 断点续传 ====================

def load_completed_uuids(output_path: str) -> set:
    """扫描已有输出文件，收集已完成的 uuid 集合（用于断点续传）

    Args:
        output_path: 输出文件路径

    Returns:
        已完成的 uuid 集合
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
    """加载元数据，跳过已完成的样本

    Args:
        metadata_path: metadata.jsonl 路径
        completed_uuids: 已完成的 uuid 集合

    Returns:
        (待处理的样本列表, 总样本数)
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


# ==================== 补足机制 ====================

def load_retry_uuids(retry_path: str) -> set:
    """从 rejected_uuids.jsonl 中加载需要重新生成的 UUID 集合

    Args:
        retry_path: rejected_uuids.jsonl 路径

    Returns:
        需要重新生成的 UUID 集合
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
    """从输出文件中删除指定 UUID 的旧记录

    使用临时文件原子替换，避免数据损坏。

    Args:
        output_path: 输出文件路径
        uuids_to_remove: 要删除的 UUID 集合

    Returns:
        删除的记录数
    """
    if not os.path.exists(output_path):
        return 0

    removed = 0
    # 写入临时文件，然后原子替换
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
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return removed


def load_retry_items(metadata_path: str, retry_uuids: set) -> list:
    """从 metadata.jsonl 中找到需要重新生成的样本

    Args:
        metadata_path: metadata.jsonl 路径
        retry_uuids: 需要重新生成的 UUID 集合

    Returns:
        待重新生成的样本列表
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


# ==================== 输入预处理 ====================

def preprocess_input(input_text: str, domain: str) -> str:
    """预处理输入文本，替换特定的提示语

    Args:
        input_text: 原始输入文本
        domain: 数据域 (math 或 science)

    Returns:
        处理后的输入文本
    """
    if domain == "math":
        # Math 类数据：替换结尾的提示语
        pattern = r"Let's think step by step and output the final answer within \\boxed\{\}\.\s*$"
        if re.search(pattern, input_text):
            input_text = re.sub(pattern, MATH_INPUT_PROMPT, input_text)
    elif domain == "science":
        # Science 类数据：替换开头的提示语
        pattern = r"^\s*Solve the following problem\. Make sure to put the answer \(and only answer\) inside \\boxed\{\}\.\s*"
        if re.search(pattern, input_text):
            input_text = re.sub(pattern, SCIENCE_INPUT_PROMPT + " ", input_text)

    return input_text


# ==================== API 调用 ====================

def extract_boxed_answer(text: str) -> str | None:
    """从文本中提取 \\boxed{...} 内的答案（栈匹配，支持嵌套大括号）

    Args:
        text: 包含 \\boxed{} 的文本

    Returns:
        提取的答案字符串，未找到则返回 None
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
    """启发式提取答案（当 \\boxed{} 不存在时）

    Args:
        text: 模型生成的文本
        domain: 数据域

    Returns:
        提取的答案，未找到返回 None
    """
    # science: 尝试匹配选择题字母
    if domain == "science":
        m = re.search(r'\b([A-E])\b\s*$', text.strip())
        if m:
            return m.group(1).upper()

    # 尝试 "answer is X" 模式
    for pat in [
        r'(?:the\s+)?answer\s+is\s*[:\s]*(.+?)(?:\.|$)',
        r'(?:therefore|thus|hence)\s*,?\s*(.+?)(?:\.|$)',
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # 如果文本很短，整段可能就是答案
    stripped = text.strip()
    if len(stripped) <= 50:
        return stripped

    return None


def extract_generated_answer(text: str, domain: str) -> str | None:
    """从生成文本中提取答案

    Args:
        text: 模型生成的文本
        domain: 数据域

    Returns:
        提取的答案，未找到返回 None
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
    """调用 vLLM OpenAI-compatible chat API（单条请求）

    Args:
        session: aiohttp 会话
        api_url: API base URL (如 http://localhost:8000/v1)
        model_name: 模型名称
        user_message: 用户消息
        system_prompt: 系统提示
        max_tokens: 最大生成 token 数
        temperature: 采样温度
        timeout: 请求超时时间（秒）
        semaphore: 并发控制信号量

    Returns:
        (生成的文本, 错误信息字典)
        成功: (text, None)
        失败: (None, {"error_type": ..., "error_message": ..., "details": ...})
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
        # 关闭 Qwen3 的内置 thinking 模式
        "chat_template_kwargs": {"enable_thinking": False},
    }

    last_error = None
    async with semaphore:
        for attempt in range(3):  # 最多重试 3 次
            try:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"]

                        # 检查 content 是否为 None
                        if content is None:
                            last_error = {
                                "error_type": "api_error",
                                "error_message": "API 返回的 content 为 None",
                                "details": {
                                    "finish_reason": data["choices"][0].get("finish_reason", ""),
                                    "attempts": attempt + 1,
                                }
                            }
                            if attempt < 2:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return (None, last_error)

                        # 检查是否被截断
                        finish_reason = data["choices"][0].get("finish_reason", "")
                        usage = data.get("usage", {})
                        completion_tokens = usage.get("completion_tokens", 0)

                        if finish_reason == "length":
                            # 输出被截断
                            return (content, {
                                "error_type": "truncated",
                                "error_message": "输出被截断（达到 max_tokens 限制）",
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
                            "error_message": f"API 返回错误状态码 {resp.status}",
                            "details": {
                                "status_code": resp.status,
                                "response_text": error_text[:500],
                                "attempts": attempt + 1,
                            }
                        }
                        if attempt < 2:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        print(f"  API 错误 (status={resp.status}): {error_text[:200]}")
                        return (None, last_error)
            except asyncio.TimeoutError as e:
                last_error = {
                    "error_type": "timeout",
                    "error_message": f"请求超时（{timeout}秒）",
                    "details": {
                        "timeout_seconds": timeout,
                        "attempts": attempt + 1,
                    }
                }
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                print(f"  请求超时: {e}")
                return (None, last_error)
            except aiohttp.ClientError as e:
                last_error = {
                    "error_type": "network_error",
                    "error_message": f"网络错误: {type(e).__name__}",
                    "details": {
                        "exception": str(e),
                        "attempts": attempt + 1,
                    }
                }
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                print(f"  网络异常: {e}")
                return (None, last_error)
            except Exception as e:
                last_error = {
                    "error_type": "unknown_error",
                    "error_message": f"未知错误: {type(e).__name__}",
                    "details": {
                        "exception": str(e),
                        "attempts": attempt + 1,
                    }
                }
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                print(f"  未知异常: {e}")
                return (None, last_error)

    return (None, last_error)


# ==================== 批量生成 ====================

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
    """异步批量生成 /no_think 回答

    Args:
        items: 待处理样本列表
        api_url: vLLM API URL
        model_name: 模型名称
        system_prompt: 系统提示
        max_tokens: 最大生成 token 数
        temperature: 采样温度
        timeout: 请求超时时间（秒）
        batch_size: 并发请求数
        output_path: 输出文件路径
        failed_output_path: 失败样本详情输出路径（JSON 格式）
        rejected_output_path: 被拒绝 UUID 输出路径（JSONL 格式，供 retry 使用）
        is_retry_mode: 是否为 retry 模式
        retry_uuids: retry 模式下要重新生成的 UUID 集合
    """
    semaphore = asyncio.Semaphore(batch_size)
    completed = 0
    failed = 0
    truncated = 0
    start_time = time.time()

    # 以追加模式打开输出文件（支持断点续传）
    f_out = open(output_path, "a", encoding="utf-8")

    # 收集失败样本
    failed_samples = []

    async with aiohttp.ClientSession() as session:

        async def process_item(item):
            nonlocal completed, failed, truncated
            uuid = item.get("uuid", "")
            input_text = item.get("input", "")
            domain = item.get("domain", "")
            ground_truth = item.get("ground_truth")

            # 预处理输入文本（替换特定提示语）
            processed_input = preprocess_input(input_text, domain)

            result, error_info = await call_chat_api(
                session, api_url, model_name, processed_input,
                system_prompt, max_tokens, temperature, timeout, semaphore,
            )

            if result is not None:
                # 从生成的输出中提取答案
                extracted_answer = extract_generated_answer(result, domain)

                output_item = {
                    "uuid": uuid,
                    "input": processed_input,  # 保存预处理后的输入
                    "domain": domain,
                    "generated_output": result,
                    "extracted_answer": extracted_answer,  # 新增：提取的答案
                    "ground_truth": ground_truth,
                }
                f_out.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                f_out.flush()
                completed += 1

                # 检查是否被截断
                if error_info and error_info.get("error_type") == "truncated":
                    truncated += 1
                    failed_samples.append({
                        "uuid": uuid,
                        "input": processed_input,  # 保存预处理后的输入
                        "domain": domain,
                        "ground_truth": ground_truth,
                        "generated_output": result,
                        "extracted_answer": extracted_answer,
                        **error_info,
                    })
            else:
                failed += 1
                # 记录失败样本
                failed_samples.append({
                    "uuid": uuid,
                    "input": processed_input,  # 保存预处理后的输入
                    "domain": domain,
                    "ground_truth": ground_truth,
                    **(error_info or {"error_type": "unknown", "error_message": "未知错误"}),
                })

            # 进度日志（每 1000 条）
            total_done = completed + failed
            if total_done % 1000 == 0 and total_done > 0:
                elapsed = time.time() - start_time
                speed = total_done / elapsed
                remaining = (len(items) - total_done) / max(speed, 0.01)
                print(
                    f"  进度: {total_done}/{len(items)} "
                    f"({total_done/len(items)*100:.1f}%) | "
                    f"成功: {completed} | 失败: {failed} | 截断: {truncated} | "
                    f"速度: {speed:.1f} 条/秒 | "
                    f"预计剩余: {remaining/60:.1f} 分钟"
                )

        # 创建所有任务并并发执行
        tasks = [process_item(item) for item in items]
        await asyncio.gather(*tasks)

    f_out.close()

    # 处理失败样本记录（retry 模式需要合并旧记录）
    if is_retry_mode and retry_uuids:
        # Retry 模式：加载旧的失败记录，删除 retry 的 UUID，然后合并新的失败记录
        old_failed_samples = []
        if os.path.exists(failed_output_path):
            with open(failed_output_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                old_failed_samples = old_data.get("failed_samples", [])

        # 删除要 retry 的 UUID
        old_failed_samples = [s for s in old_failed_samples if s.get("uuid") not in retry_uuids]

        # 合并新的失败记录
        all_failed_samples = old_failed_samples + failed_samples

        # 同样处理 rejected_uuids.jsonl
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
        # 正常模式：直接使用新的失败记录
        all_failed_samples = failed_samples
        old_rejected_items = []

    # 保存失败样本到 JSON 文件（详细信息）
    if all_failed_samples:
        # 统计错误类型
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

        # 保存 rejected_uuids.jsonl（供 retry 使用）
        with open(rejected_output_path, "w", encoding="utf-8") as f_rejected:
            # 先写入旧的记录
            for item in old_rejected_items:
                f_rejected.write(json.dumps(item, ensure_ascii=False) + "\n")
            # 再写入新的失败记录
            for sample in failed_samples:
                rejected_item = {
                    "uuid": sample.get("uuid", ""),
                    "domain": sample.get("domain", "unknown"),
                    "reason": f"generation_failed_{sample.get('error_type', 'unknown')}",
                    "input": sample.get("input", ""),
                }
                f_rejected.write(json.dumps(rejected_item, ensure_ascii=False) + "\n")

    # 最终统计
    elapsed = time.time() - start_time
    print(f"\n  生成完成!")
    print(f"  成功: {completed} | 失败: {failed} | 截断: {truncated}")
    print(f"  总耗时: {elapsed/60:.1f} 分钟")
    print(f"  平均速度: {(completed + failed) / max(elapsed, 0.01):.1f} 条/秒")
    if all_failed_samples:
        print(f"  失败详情已保存到: {failed_output_path}")
        print(f"  被拒绝 UUID 已保存到: {rejected_output_path}")
        if is_retry_mode:
            print(f"  (Retry 模式: 已更新失败记录，移除成功的 UUID)")


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(
        description="通过 vLLM API 批量生成 /no_think 回答（仅 math + science）"
    )
    parser.add_argument(
        "--metadata_input",
        type=str,
        default="datasets/superior_reasoning/metadata.jsonl",
        help="元数据文件路径（默认: datasets/superior_reasoning/metadata.jsonl）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/superior_reasoning/no_think_raw.jsonl",
        help="输出文件路径（默认: datasets/superior_reasoning/no_think_raw.jsonl）",
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API URL（默认: http://localhost:8000/v1）",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen3-32B",
        help="模型名称（默认: Qwen3-32B）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="并发请求数（默认: 256）",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="最大生成 token 数（默认: 512）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="采样温度（默认: 0.0，贪心解码）",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="单个请求超时时间（秒，默认: 300）",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="自定义 system prompt（默认使用内置 prompt）",
    )
    parser.add_argument(
        "--retry_from",
        type=str,
        default=None,
        help="rejected_uuids.jsonl 路径，仅重新生成被拒绝的样本",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅处理前 N 条样本（0=不限制，用于冒烟测试）",
    )
    parser.add_argument(
        "--failed_output",
        type=str,
        default=None,
        help="失败样本详情输出路径（JSON 格式，默认自动生成）",
    )
    parser.add_argument(
        "--rejected_output",
        type=str,
        default=None,
        help="被拒绝 UUID 输出路径（JSONL 格式，默认自动生成，供 retry 使用）",
    )
    args = parser.parse_args()

    # 解析路径
    metadata_path = project_root / args.metadata_input
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 失败样本详情输出路径（默认与输出文件同目录，_failed.json 后缀）
    if args.failed_output:
        failed_output_path = project_root / args.failed_output
    else:
        failed_output_path = output_path.parent / (output_path.stem + "_failed.json")

    # 被拒绝 UUID 输出路径（默认为 rejected_uuids.jsonl）
    if args.rejected_output:
        rejected_output_path = project_root / args.rejected_output
    else:
        rejected_output_path = output_path.parent / "rejected_uuids.jsonl"

    system_prompt = args.system_prompt if args.system_prompt else SYSTEM_PROMPT

    print(f"\n{'='*60}")
    print("Step 2: 生成 /no_think 回答")
    print(f"{'='*60}")
    print(f"元数据文件: {metadata_path}")
    print(f"输出文件:   {output_path}")
    print(f"失败详情:   {failed_output_path}")
    print(f"被拒绝UUID: {rejected_output_path}")
    print(f"API URL:    {args.api_url}")
    print(f"模型:       {args.model_name}")
    print(f"并发数:     {args.batch_size}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"temperature: {args.temperature}")
    print(f"timeout:    {args.timeout} 秒")
    if args.retry_from:
        print(f"补足模式:   从 {args.retry_from} 读取被拒绝的 UUID")
    if args.limit > 0:
        print(f"限制数量:   仅处理前 {args.limit} 条")

    if args.retry_from:
        # ---- 补足模式 ----
        retry_path = project_root / args.retry_from
        print(f"\n{'─'*40}")
        print("补足模式: 重新生成被拒绝的样本")

        # 1. 读取被拒绝的 UUID
        retry_uuids = load_retry_uuids(str(retry_path))
        print(f"  被拒绝的 UUID 数: {len(retry_uuids)}")

        if not retry_uuids:
            print("  无需补足，退出。")
            return

        # 2. 从输出文件中删除旧记录
        removed = remove_uuids_from_output(str(output_path), retry_uuids)
        print(f"  从输出文件中删除旧记录: {removed} 条")

        # 3. 从 metadata 中找到对应样本
        pending = load_retry_items(str(metadata_path), retry_uuids)
        print(f"  找到待重新生成的样本: {len(pending)} 条")

        if not pending:
            print("  在 metadata 中未找到对应样本，退出。")
            return

        # 标记为 retry 模式
        is_retry_mode = True
    else:
        # ---- 正常模式 ----
        # 1. 加载已完成的 uuid（断点续传）
        print(f"\n{'─'*40}")
        print("1. 检查断点续传状态...")
        completed_uuids = load_completed_uuids(str(output_path))
        if completed_uuids:
            print(f"  发现 {len(completed_uuids)} 条已完成记录，将跳过这些样本")
        else:
            print("  无已完成记录，从头开始")

        # 2. 加载待处理样本
        print(f"\n{'─'*40}")
        print("2. 加载待处理样本...")
        pending, total = load_metadata(str(metadata_path), completed_uuids)
        print(f"  元数据总数: {total}")
        print(f"  待处理:     {len(pending)}")
        print(f"  已跳过:     {total - len(pending)}")

        # 标记为正常模式
        is_retry_mode = False
        retry_uuids = set()

    # 应用 --limit
    if args.limit > 0 and len(pending) > args.limit:
        pending = pending[:args.limit]
        print(f"  应用 --limit={args.limit}，实际处理: {len(pending)} 条")

    if not pending:
        print("\n  所有样本已处理完毕，无需生成。")
        return

    # ---- 测试 API 连通性 ----
    print(f"\n{'─'*40}")
    print("测试 API 连通性...")
    import urllib.request
    try:
        req = urllib.request.Request(f"{args.api_url}/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"  API 连通正常 (status={resp.status})")
    except Exception as e:
        print(f"  警告: API 连通测试失败: {e}")
        print(f"  将继续尝试生成，但可能会遇到错误...")

    # ---- 开始批量生成 ----
    print(f"\n{'─'*40}")
    print("开始批量生成...")
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
    print("Step 2 完成!")
    print(f"输出文件: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
