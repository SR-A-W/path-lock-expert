#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""过滤生成的 /no_think 数据

合并两个来源的 /no_think 数据并进行三重过滤：
  1. 正确性过滤：与 ground truth 对比答案是否正确（仅 math+science）
  2. 风格过滤：检测是否包含推理行为（反思性词汇）
  3. 长度过滤：输出 token 数不能过长

输入：
  - no_think_raw.jsonl（math+science，模型生成）
  - no_think_instruction.jsonl（instruction，直接提取）

输出：
  - no_think_filtered.jsonl — 通过过滤的样本（ShareGPT 格式）
  - rejected_uuids.jsonl   — 被拒绝的样本（UUID + 拒绝原因），供 Step 2 补足
    追加模式：Step 2 已写入生成失败的样本，Step 3 追加过滤拒绝的样本

使用方法:
    conda activate het
    python -m src.datasets.filter_no_think_data \
        --raw_input datasets/superior_reasoning/no_think_raw.jsonl \
        --instruction_input datasets/superior_reasoning/no_think_instruction.jsonl \
        --output datasets/superior_reasoning/no_think_filtered.jsonl \
        --rejected_output datasets/superior_reasoning/rejected_uuids.jsonl

依赖:
    pip install transformers  (用于 token 计数)
"""

import argparse
import asyncio
import json
import math
import re
import sys
from fractions import Fraction
from pathlib import Path

import aiohttp

# 项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ==================== Token 计数 ====================

# 全局 tokenizer（延迟加载）
_tokenizer = None


def get_tokenizer():
    """延迟加载 tokenizer（Qwen2.5 tokenizer，用于 token 计数）"""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        # 使用 Qwen2.5 的 tokenizer 进行 token 计数
        # 如果本地没有，会自动下载
        _tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            trust_remote_code=True,
        )
    return _tokenizer


def count_tokens(text: str) -> int:
    """计算文本的 token 数

    Args:
        text: 输入文本

    Returns:
        token 数量
    """
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


# ==================== 答案提取 ====================

def extract_boxed_answer(text: str) -> str | None:
    """从文本中提取 \\boxed{...} 内的答案（栈匹配，支持嵌套大括号）

    如果有多个 \\boxed{}，取最后一个。

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


def extract_answer_from_text(text: str, domain: str) -> str | None:
    """从生成文本中提取答案（先尝试 \\boxed{}，再用启发式）

    Args:
        text: 模型生成的文本
        domain: 数据域

    Returns:
        提取的答案，未找到返回 None
    """
    # 优先 \boxed{}
    answer = extract_boxed_answer(text)
    if answer is not None:
        return answer

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

    # 最后手段：如果文本很短，整段可能就是答案
    stripped = text.strip()
    if len(stripped) <= 50:
        return stripped

    return None


# ==================== 答案归一化 ====================

def normalize_latex(text: str) -> str:
    """清理 LaTeX 格式，便于比较

    Args:
        text: 含 LaTeX 的字符串

    Returns:
        归一化后的字符串
    """
    s = text.strip()

    # 移除 displaystyle, textstyle 等样式命令
    s = re.sub(r'\\(?:displaystyle|textstyle|scriptstyle|scriptscriptstyle)\s*', '', s)

    # 统一分数命令：\dfrac, \tfrac, \cfrac → \frac
    s = re.sub(r'\\[dtc]frac\b', r'\\frac', s)

    # 移除 LaTeX 空格命令：\,  \;  \:  \!  \quad  \qquad  等
    s = re.sub(r'\\[,;:!\s]', '', s)
    s = re.sub(r'\\(?:quad|qquad|enspace|thinspace|medspace|thickspace)\b', '', s)

    # 移除 \text{}, \mathrm{}, \textbf{}, \mathbf{} 等包装
    s = re.sub(r'\\(?:text|mathrm|textbf|mathbf|textrm|textit|mathit|mathcal|mathbb|mathfrak)\s*\{([^}]*)\}', r'\1', s)

    # 移除 \left, \right
    s = re.sub(r'\\(?:left|right)\s*', '', s)

    # 移除多余空格
    s = re.sub(r'\s+', ' ', s).strip()

    return s


def try_parse_number(text: str) -> float | None:
    """尝试将文本解析为数值（支持分数、百分比、LaTeX 分数）

    Args:
        text: 可能是数值的字符串

    Returns:
        解析出的浮点数，失败返回 None
    """
    s = text.strip()
    s_no_comma = s.replace(',', '')

    # 直接尝试 float
    try:
        return float(s_no_comma)
    except ValueError:
        pass

    # 尝试 Python 分数（如 1/2）
    try:
        return float(Fraction(s_no_comma))
    except (ValueError, ZeroDivisionError):
        pass

    # 尝试 LaTeX 分数 \frac{a}{b}
    m = re.match(r'\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}', s)
    if m:
        try:
            num = float(m.group(1))
            den = float(m.group(2))
            if den != 0:
                return num / den
        except ValueError:
            pass

    # 百分比
    m = re.match(r'([\d.]+)\s*%', s)
    if m:
        try:
            return float(m.group(1)) / 100.0
        except ValueError:
            pass

    return None


def normalize_answer(answer: str) -> str:
    """归一化答案用于比较

    Args:
        answer: 原始答案字符串

    Returns:
        归一化后的答案字符串
    """
    cleaned = normalize_latex(answer)

    # 尝试数值化
    num = try_parse_number(cleaned)
    if num is not None and not (math.isinf(num) or math.isnan(num)):
        if num == int(num):
            return str(int(num))
        return f"{num:.10g}"

    # 字符串归一化：小写、去空格、去标点
    s = cleaned.lower().strip()
    s = re.sub(r'[.,;:!?\s]+', '', s)
    return s


# ==================== 过滤器 ====================

# 推理行为指示词模式（大小写不敏感匹配）
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
# 预编译正则以提高性能
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in REASONING_PATTERNS]


# Model-based 判断的 prompt
MODEL_JUDGE_PROMPT = """You are a mathematical equivalence checker. Your task is to determine if two mathematical answers are equivalent.

**Problem:**
{problem}

**Model's Complete Response:**
{full_response}

**Extracted Answer from Model:**
{generated_answer}

**Ground Truth Answer:**
{ground_truth}

Your task: Determine if the "Extracted Answer from Model" and "Ground Truth Answer" are mathematically equivalent.

Consider:
1. Different LaTeX formatting (e.g., \\frac vs \\dfrac, \\displaystyle)
2. Different variable names or representations (e.g., x=1, y=12 vs 1³+12³)
3. Approximate vs exact values (e.g., \\approx vs exact number)
4. Different but equivalent mathematical expressions
5. Use the full response and problem context to understand the answer better

Output your judgment in \\boxed{{YES}} if they are equivalent, or \\boxed{{NO}} if they are not equivalent.

Only output \\boxed{{YES}} or \\boxed{{NO}}, nothing else."""


async def check_correctness_with_model(
    generated_answer: str,
    ground_truth: str,
    full_response: str,
    problem: str,
    api_url: str,
    model_name: str,
    timeout: int = 60,
) -> bool:
    """使用模型判断答案是否等价

    Args:
        generated_answer: 提取的生成答案
        ground_truth: 标准答案
        full_response: 模型的完整回复
        problem: 问题描述（题干）
        api_url: vLLM API URL
        model_name: 模型名称
        timeout: 超时时间（秒）

    Returns:
        True 表示等价，False 表示不等价
    """
    prompt = MODEL_JUDGE_PROMPT.format(
        problem=problem,
        full_response=full_response,
        generated_answer=generated_answer,
        ground_truth=ground_truth,
    )

    url = f"{api_url}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 50,
        "temperature": 0.0,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]

                    # 提取 \boxed{} 中的内容
                    match = re.search(r'\\boxed\s*\{([^}]+)\}', content)
                    if match:
                        result = match.group(1).strip().upper()
                        return result == "YES"

                    # 如果没有 \boxed{}，尝试直接匹配 YES/NO
                    if "YES" in content.upper():
                        return True
                    if "NO" in content.upper():
                        return False

                    # 无法判断，默认返回 False
                    return False
                else:
                    # API 错误，默认返回 False
                    return False
    except Exception:
        # 任何异常，默认返回 False
        return False


def check_correctness(
    generated: str,
    ground_truth: str | None,
    domain: str,
    input_text: str = "",
    use_model: bool = False,
    api_url: str | None = None,
    model_name: str | None = None,
) -> bool:
    """过滤器 1：正确性检查

    从生成文本中提取答案，与 ground truth 归一化后比较。
    如果 ground_truth 为 None，跳过检查（返回 True）。

    可选：使用模型进行等价性判断（处理复杂的数学等价情况）

    Args:
        generated: 模型生成的文本
        ground_truth: 标准答案
        domain: 数据域
        input_text: 问题描述（题干）
        use_model: 是否使用模型判断（可选）
        api_url: vLLM API URL（use_model=True 时需要）
        model_name: 模型名称（use_model=True 时需要）

    Returns:
        True 表示通过（答案正确或无法检查）
    """
    if ground_truth is None:
        return True

    gen_answer = extract_answer_from_text(generated, domain)
    if gen_answer is None:
        return False

    # 先尝试基于规则的归一化比较
    if normalize_answer(gen_answer) == normalize_answer(ground_truth):
        return True

    # 如果规则判断失败，且启用了模型判断，则使用模型
    if use_model and api_url and model_name:
        # 注意：这是同步函数，但调用了异步函数
        # 需要在调用处使用 asyncio.run()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            check_correctness_with_model(
                generated_answer=gen_answer,
                ground_truth=ground_truth,
                full_response=generated,
                problem=input_text,
                api_url=api_url,
                model_name=model_name,
            )
        )

    return False


def check_no_reasoning_style(text: str) -> bool:
    """过滤器 2：风格检查（无推理行为）

    检测文本中是否包含推理/反思性词汇。

    Args:
        text: 模型生成的文本

    Returns:
        True 表示通过（不含推理行为）
    """
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(text):
            return False
    return True


def check_token_length(text: str, max_tokens: int) -> bool:
    """过滤器 3：token 长度检查

    Args:
        text: 模型生成的文本
        max_tokens: 最大 token 数

    Returns:
        True 表示通过（token 数在阈值内）
    """
    return count_tokens(text.strip()) <= max_tokens


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(
        description="过滤 /no_think 数据（正确性 + 风格 + token 长度）"
    )
    parser.add_argument(
        "--raw_input",
        type=str,
        default="datasets/superior_reasoning/no_think_raw.jsonl",
        help="模型生成的原始 /no_think 数据路径（math+science）",
    )
    parser.add_argument(
        "--instruction_input",
        type=str,
        default="datasets/superior_reasoning/no_think_instruction.jsonl",
        help="instruction domain 直接提取的 /no_think 数据路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="过滤后输出路径（默认：与 raw_input 同目录，文件名为 *_filtered.jsonl）",
    )
    parser.add_argument(
        "--rejected_output",
        type=str,
        default=None,
        help="被拒绝样本的 UUID 列表输出路径（默认：与 raw_input 同目录，文件名为 rejected_uuids.jsonl）",
    )
    parser.add_argument("--max_tokens_math", type=int, default=800, help="math 最大 token 数（默认: 800）")
    parser.add_argument("--max_tokens_science", type=int, default=800, help="science 最大 token 数（默认: 800）")
    parser.add_argument("--max_tokens_instruction", type=int, default=500, help="instruction 最大 token 数（默认: 500）")
    parser.add_argument("--skip_correctness", action="store_true", help="跳过正确性过滤")
    parser.add_argument("--skip_style", action="store_true", help="跳过风格过滤")
    parser.add_argument("--skip_length", action="store_true", help="跳过长度过滤")
    parser.add_argument(
        "--use_model_judge",
        action="store_true",
        help="使用模型判断答案等价性（可选，用于处理复杂的数学等价情况）",
    )
    parser.add_argument(
        "--judge_api_url",
        type=str,
        default="http://localhost:8000/v1",
        help="模型判断 API URL（use_model_judge=True 时需要）",
    )
    parser.add_argument(
        "--judge_model_name",
        type=str,
        default="Qwen3-32B",
        help="模型判断使用的模型名称（use_model_judge=True 时需要）",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="输出文件名后缀（如 retry_20260313_120000），用于避免覆盖已有文件",
    )
    args = parser.parse_args()

    raw_input_path = project_root / args.raw_input
    instruction_input_path = project_root / args.instruction_input

    # 构造后缀字符串
    suffix_str = f"_{args.suffix}" if args.suffix else ""

    # 输出路径：默认与 raw_input 同目录，文件名为 *_filtered[_suffix].jsonl
    if args.output:
        output_path = project_root / args.output
    else:
        output_path = raw_input_path.parent / (raw_input_path.stem + f"_filtered{suffix_str}.jsonl")

    # rejected 输出路径：默认与 raw_input 同目录，文件名为 rejected_uuids[_suffix].jsonl
    if args.rejected_output:
        rejected_path = project_root / args.rejected_output
    else:
        rejected_path = raw_input_path.parent / f"rejected_uuids{suffix_str}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_tokens = {
        "math": args.max_tokens_math,
        "science": args.max_tokens_science,
        "instruction-following": args.max_tokens_instruction,
    }

    print(f"\n{'='*60}")
    print("Step 3: 过滤 /no_think 数据")
    print(f"{'='*60}")
    print(f"原始输入 (math+science): {raw_input_path}")
    print(f"指令输入 (instruction):  {instruction_input_path}")
    print(f"输出: {output_path}")
    print(f"被拒绝输出: {rejected_path} (追加模式)")
    print(f"token 长度阈值: {max_tokens}")
    print(f"跳过正确性: {args.skip_correctness}")
    print(f"跳过风格:   {args.skip_style}")
    print(f"跳过长度:   {args.skip_length}")
    if args.use_model_judge:
        print(f"模型判断:   启用")
        print(f"  API URL:  {args.judge_api_url}")
        print(f"  模型:     {args.judge_model_name}")
    else:
        print(f"模型判断:   禁用（仅使用规则判断）")

    # ---- 1. 加载数据 ----
    print(f"\n{'─'*40}")
    print("1. 加载数据...")

    # 加载 math+science 原始生成数据
    raw_items = []
    if raw_input_path.exists():
        with open(raw_input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_items.append(json.loads(line))
    print(f"  math+science 原始数据: {len(raw_items)} 条")

    # 加载 instruction 直接提取数据
    instruction_items = []
    if instruction_input_path.exists():
        with open(instruction_input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    instruction_items.append(json.loads(line))
    print(f"  instruction 数据: {len(instruction_items)} 条")

    # ---- 2. 过滤 math+science ----
    print(f"\n{'─'*40}")
    print("2. 过滤 math+science 数据...")

    # 输出长度分布（辅助调参）
    if raw_items:
        lengths = [count_tokens(it.get("generated_output", "").strip()) for it in raw_items[:1000]]
        if lengths:
            lengths_sorted = sorted(lengths)
            n = len(lengths_sorted)
            print(f"\n  输出 token 长度分布（前 {n} 条采样）:")
            for p in [25, 50, 75, 90, 95, 99]:
                idx = min(int(n * p / 100), n - 1)
                print(f"    P{p}: {lengths_sorted[idx]} tokens")

    # 统计计数器
    total = 0
    pass_all = 0
    domain_stats = {}
    filtered_items = []
    rejected_items = []

    for item in raw_items:
        total += 1
        domain = item.get("domain", "unknown")
        generated = item.get("generated_output", "")
        ground_truth = item.get("ground_truth")
        uuid = item.get("uuid", "")
        input_text = item.get("input", "")

        # 初始化 domain 统计
        if domain not in domain_stats:
            domain_stats[domain] = {
                "total": 0, "pass_correctness": 0,
                "pass_style": 0, "pass_length": 0, "pass_all": 0,
            }
        domain_stats[domain]["total"] += 1

        # 过滤器 1: 正确性
        ok_correct = True if args.skip_correctness else check_correctness(
            generated, ground_truth, domain,
            input_text=input_text,
            use_model=args.use_model_judge,
            api_url=args.judge_api_url if args.use_model_judge else None,
            model_name=args.judge_model_name if args.use_model_judge else None,
        )
        if ok_correct:
            domain_stats[domain]["pass_correctness"] += 1

        # 过滤器 2: 风格
        ok_style = True if args.skip_style else check_no_reasoning_style(generated)
        if ok_style:
            domain_stats[domain]["pass_style"] += 1

        # 过滤器 3: token 长度
        domain_max = max_tokens.get(domain, 800)
        ok_length = True if args.skip_length else check_token_length(generated, domain_max)
        if ok_length:
            domain_stats[domain]["pass_length"] += 1

        # 全部通过
        if ok_correct and ok_style and ok_length:
            pass_all += 1
            domain_stats[domain]["pass_all"] += 1
            filtered_items.append(item)
        else:
            # 记录拒绝原因（取第一个失败的原因）
            if not ok_correct:
                reason = "correctness"
            elif not ok_style:
                reason = "style"
            else:
                reason = "length"
            rejected_items.append({
                "uuid": uuid,
                "domain": domain,
                "reason": reason,
                "input": input_text,
            })

        # 进度
        if total % 50000 == 0:
            print(f"  已处理: {total}/{len(raw_items)}...")

    # ---- 3. 过滤 instruction ----
    print(f"\n{'─'*40}")
    print("3. 过滤 instruction 数据...")

    instruction_pass = 0
    instruction_total = 0
    if "instruction-following" not in domain_stats:
        domain_stats["instruction-following"] = {
            "total": 0, "pass_correctness": 0,
            "pass_style": 0, "pass_length": 0, "pass_all": 0,
        }

    for item in instruction_items:
        instruction_total += 1
        domain_stats["instruction-following"]["total"] += 1
        uuid = item.get("uuid", "")
        domain = item.get("domain", "instruction-following")

        # 从 ShareGPT 格式中提取 assistant 回答
        conversations = item.get("conversations", [])
        assistant_text = ""
        input_text = ""
        for conv in conversations:
            if conv.get("from") == "assistant":
                assistant_text = conv.get("value", "")
            elif conv.get("from") == "user":
                input_text = conv.get("value", "")

        # instruction 不做正确性检查
        domain_stats["instruction-following"]["pass_correctness"] += 1

        # 风格过滤
        ok_style = True if args.skip_style else check_no_reasoning_style(assistant_text)
        if ok_style:
            domain_stats["instruction-following"]["pass_style"] += 1

        # token 长度过滤
        inst_max = max_tokens.get("instruction-following", 500)
        ok_length = True if args.skip_length else check_token_length(assistant_text, inst_max)
        if ok_length:
            domain_stats["instruction-following"]["pass_length"] += 1

        if ok_style and ok_length:
            instruction_pass += 1
            domain_stats["instruction-following"]["pass_all"] += 1
            # 已经是 ShareGPT 格式，直接保留
            filtered_items.append(item)
        else:
            reason = "style" if not ok_style else "length"
            rejected_items.append({
                "uuid": uuid,
                "domain": domain,
                "reason": reason,
                "input": input_text,
            })

    print(f"  instruction 通过: {instruction_pass}/{instruction_total}")

    # ---- 4. 写入过滤后的数据 ----
    print(f"\n{'─'*40}")
    print("4. 写入过滤后的数据...")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in filtered_items:
            # math+science 需要转换为 ShareGPT 格式
            if "generated_output" in item:
                sharegpt = {
                    "conversations": [
                        {"from": "user", "value": f"{item['input']}/no_think"},
                        {"from": "assistant", "value": item["generated_output"]},
                    ]
                }
                f.write(json.dumps(sharegpt, ensure_ascii=False) + "\n")
            else:
                # instruction 已经是 ShareGPT 格式（含 uuid, domain 等额外字段）
                # 只保留 conversations 字段
                sharegpt = {"conversations": item["conversations"]}
                f.write(json.dumps(sharegpt, ensure_ascii=False) + "\n")

    print(f"  写入 {len(filtered_items)} 条")

    # ---- 5. 写入被拒绝的 UUID 到 rejected_uuids.jsonl ----
    print(f"\n{'─'*40}")
    print("5. 写入被拒绝的 UUID...")

    # 幂等写入：先加载已有的 Step 2 条目（generation_failed_* 原因），保留它们
    # Step 3 的条目每次重新写入，重复运行不会产生重复 UUID
    existing_step2_items = []
    if rejected_path.exists():
        with open(rejected_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    # 保留 Step 2 写入的生成失败条目（reason 以 generation_failed 开头）
                    if entry.get("reason", "").startswith("generation_failed"):
                        existing_step2_items.append(entry)
        print(f"  保留 Step 2 生成失败条目: {len(existing_step2_items)} 条")

    # 写模式：Step 2 条目 + Step 3 新条目
    with open(rejected_path, "w", encoding="utf-8") as f:
        for item in existing_step2_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        for item in rejected_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  写入 {len(existing_step2_items)} 条 Step 2 条目 + {len(rejected_items)} 条 Step 3 过滤拒绝记录")

    # ---- 6. 输出统计报告 ----
    print(f"\n{'─'*40}")
    print("6. 过滤统计报告")

    total_all = total + instruction_total
    total_pass = pass_all + instruction_pass
    print(f"\n  总样本数: {total_all}")
    print(f"  全部通过: {total_pass}/{total_all} ({total_pass/max(total_all,1)*100:.1f}%)")
    print(f"  被拒绝:   {len(rejected_items)}")

    print(f"\n  各 domain 通过率:")
    for domain, ds in sorted(domain_stats.items()):
        dt = ds["total"]
        dp = ds["pass_all"]
        print(f"    {domain}: {dp}/{dt} ({dp/max(dt,1)*100:.1f}%)")
        print(f"      正确性: {ds['pass_correctness']}/{dt}, "
              f"风格: {ds['pass_style']}/{dt}, "
              f"长度: {ds['pass_length']}/{dt}")

    # 拒绝原因分布
    reason_counts = {}
    for item in rejected_items:
        key = f"{item['domain']}:{item['reason']}"
        reason_counts[key] = reason_counts.get(key, 0) + 1
    if reason_counts:
        print(f"\n  拒绝原因分布:")
        for key, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"    {key}: {count}")

    print(f"\n  输出文件:")
    print(f"    过滤后数据: {output_path}")
    print(f"    被拒绝 UUID: {rejected_path}")
    print(f"\n{'='*60}")
    print("Step 3 完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
