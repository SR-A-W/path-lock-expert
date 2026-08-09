#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 Superior-Reasoning-SFT 数据集转换为 /think 训练格式

从 HuggingFace 下载 Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b 数据集，
过滤指定 domain（默认 math + science + instruction-following），转换为 LLaMA-Factory
ShareGPT 格式，同时提取 ground truth 用于后续 /no_think 数据的正确性过滤。

instruction-following domain 特殊处理：直接提取 </think> 之后的文本作为 /no_think 回答。

输出三个文件（到 datasets/superior_reasoning/）：
  1. think_data.jsonl           — 全部 domain 的 /think 训练数据（ShareGPT 格式）
  2. metadata.jsonl             — math + science 的问题 + ground truth（供 Step 2 使用）
  3. no_think_instruction.jsonl — instruction domain 直接提取的 /no_think（ShareGPT 格式）

使用方法:
    conda activate het
    python -m src.datasets.prepare_think_data \
        --output_dir datasets/superior_reasoning

依赖:
    pip install datasets
"""

import argparse
import json
import re
import sys
from pathlib import Path

# 项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ==================== Ground Truth 提取 ====================

def extract_boxed_answer(text: str) -> str | None:
    """从文本中提取 \\boxed{...} 内的答案（支持嵌套大括号）

    使用栈匹配算法处理嵌套情况，如 \\boxed{\\frac{1}{2}}。
    如果有多个 \\boxed{}，取最后一个（通常是最终答案）。

    Args:
        text: 包含 \\boxed{} 的文本

    Returns:
        提取的答案字符串，未找到则返回 None
    """
    pattern = r'\\boxed\s*\{'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    # 取最后一个 \boxed{} （通常是最终答案）
    last_match = matches[-1]
    start = last_match.end()  # { 之后的位置

    # 栈匹配：找到对应的闭合 }
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

    # 提取 {} 内的内容
    answer = text[start:pos - 1].strip()
    return answer if answer else None


def extract_final_answer_heuristic(text: str, domain: str) -> str | None:
    """启发式提取最终答案（当 \\boxed{} 不存在时的备选方案）

    Args:
        text: 模型输出文本
        domain: 数据域（math/science）

    Returns:
        提取的答案字符串，未找到则返回 None
    """
    # 尝试匹配 "the answer is X" 模式
    answer_patterns = [
        r'(?:the\s+)?answer\s+is\s*[:\s]*(.+?)(?:\.|$)',
        r'(?:the\s+)?result\s+is\s*[:\s]*(.+?)(?:\.|$)',
        r'(?:therefore|thus|hence|so)\s*,?\s*(.+?)(?:\.|$)',
    ]
    for pat in answer_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # science domain: 尝试匹配选择题答案 (A/B/C/D/E)
    if domain == "science":
        m = re.search(r'\b([A-E])\b\s*$', text.strip())
        if m:
            return m.group(1).upper()

    return None


# ==================== 格式转换 ====================

def convert_to_sharegpt_think(input_text: str, output_text: str) -> dict:
    """将单条样本转换为 ShareGPT /think 格式

    在 user message 末尾追加 /think 控制 token。

    Args:
        input_text: 用户问题
        output_text: 模型回答（含 <think> 推理过程）

    Returns:
        ShareGPT 格式的字典
    """
    return {
        "conversations": [
            {"from": "user", "value": f"{input_text}/think"},
            {"from": "assistant", "value": output_text},
        ]
    }


def convert_to_sharegpt_no_think(input_text: str, output_text: str) -> dict:
    """将单条样本转换为 ShareGPT /no_think 格式

    在 user message 末尾追加 /no_think 控制 token。

    Args:
        input_text: 用户问题
        output_text: 模型回答（简短直接）

    Returns:
        ShareGPT 格式的字典
    """
    return {
        "conversations": [
            {"from": "user", "value": f"{input_text}/no_think"},
            {"from": "assistant", "value": output_text},
        ]
    }


def extract_ground_truth(output_text: str, domain: str) -> str | None:
    """从模型输出中提取 ground truth 答案

    优先使用 \\boxed{} 提取，失败则使用启发式方法。

    Args:
        output_text: 模型输出文本
        domain: 数据域

    Returns:
        ground truth 字符串，无法提取则返回 None
    """
    answer = extract_boxed_answer(output_text)
    if answer is not None:
        return answer
    return extract_final_answer_heuristic(output_text, domain)


def extract_after_think_block(output_text: str) -> str | None:
    """从 output 中提取 </think> 之后的文本

    用于 instruction-following domain：直接提取 think block 之后的内容
    作为 /no_think 回答。

    Args:
        output_text: 模型输出文本（可能包含 <think>...</think> 块）

    Returns:
        </think> 之后的文本，如果无 think block 则返回整段文本，
        如果提取结果为空则返回 None
    """
    idx = output_text.find("</think>")
    if idx == -1:
        # 无 think block，整段就是答案
        return output_text.strip() or None
    result = output_text[idx + len("</think>"):].strip()
    return result or None


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(
        description="将 Superior-Reasoning-SFT 数据集转换为 /think 训练格式"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/superior_reasoning",
        help="输出目录（默认: datasets/superior_reasoning）",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b",
        help="HuggingFace 数据集名称",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="stage1",
        help="要包含的训练阶段，逗号分隔（默认: stage1）",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="math,science",
        help="要包含的 domain，逗号分隔（默认: math,science）",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace 数据集缓存目录",
    )
    args = parser.parse_args()

    # 解析参数
    target_stages = set(args.stages.split(","))
    target_domains = set(args.domains.split(","))
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    think_output = output_dir / "think_data.jsonl"
    metadata_output = output_dir / "metadata.jsonl"
    instruction_no_think_output = output_dir / "no_think_instruction.jsonl"

    print(f"\n{'='*60}")
    print("Step 1: 准备 /think 训练数据")
    print(f"{'='*60}")
    print(f"数据集: {args.dataset_name}")
    print(f"目标 stages: {target_stages}")
    print(f"目标 domains: {target_domains}")
    print(f"输出目录: {output_dir}")

    # ---- 1. 加载数据集 ----
    print(f"\n{'─'*40}")
    print("1. 从 HuggingFace 加载数据集...")
    from datasets import load_dataset

    # 加载多个 stage 并合并
    all_data = []
    for stage in target_stages:
        print(f"  加载 {stage}...")
        ds = load_dataset(args.dataset_name, stage, cache_dir=args.cache_dir)
        # 数据集可能有 train split 或直接是 Dataset
        if isinstance(ds, dict):
            if "train" in ds:
                stage_data = ds["train"]
            else:
                first_key = list(ds.keys())[0]
                stage_data = ds[first_key]
        else:
            stage_data = ds

        print(f"    {stage} 样本数: {len(stage_data)}")
        all_data.extend(stage_data)

    data = all_data
    print(f"  总样本数: {len(data)}")

    # ---- 2. 过滤 domain ----
    print(f"\n{'─'*40}")
    print("2. 过滤 domain...")

    # 统计各 domain 数量
    domain_counts = {}
    for item in data:
        d = item.get("domain", "unknown")
        domain_counts[d] = domain_counts.get(d, 0) + 1
    print(f"  原始 domain 分布: {domain_counts}")

    # 过滤（stage 已在加载时过滤）
    filtered = []
    for item in data:
        domain = item.get("domain", "unknown")
        if domain in target_domains:
            filtered.append(item)

    print(f"  过滤后样本数: {len(filtered)}")

    # ---- 3. 转换格式并提取 ground truth ----
    print(f"\n{'─'*40}")
    print("3. 转换格式并提取 ground truth...")

    think_count = 0
    gt_found_count = 0
    gt_missing_count = 0
    instruction_no_think_count = 0
    instruction_extract_fail_count = 0
    metadata_count = 0
    domain_stats = {}

    with open(think_output, "w", encoding="utf-8") as f_think, \
         open(metadata_output, "w", encoding="utf-8") as f_meta, \
         open(instruction_no_think_output, "w", encoding="utf-8") as f_inst:

        for item in filtered:
            uuid = item.get("uuid", "")
            input_text = item.get("input", "")
            output_text = item.get("output", "")
            domain = item.get("domain", "unknown")

            # meta 字段是 JSON 字符串，需要解析
            meta_str = item.get("meta", "{}")
            try:
                meta_dict = json.loads(meta_str) if isinstance(meta_str, str) else meta_str
                stage = meta_dict.get("training_stage", "unknown")
            except (json.JSONDecodeError, AttributeError):
                stage = "unknown"

            # 跳过空样本
            if not input_text.strip() or not output_text.strip():
                continue

            # 初始化 domain 统计
            if domain not in domain_stats:
                domain_stats[domain] = {"total": 0, "gt_found": 0, "no_think_extracted": 0}
            domain_stats[domain]["total"] += 1

            # 所有 domain 都转换为 ShareGPT /think 格式
            sharegpt = convert_to_sharegpt_think(input_text, output_text)
            f_think.write(json.dumps(sharegpt, ensure_ascii=False) + "\n")
            think_count += 1

            if domain == "instruction-following":
                # instruction-following: 直接提取 </think> 之后的文本作为 /no_think
                no_think_text = extract_after_think_block(output_text)
                if no_think_text:
                    no_think_sharegpt = convert_to_sharegpt_no_think(input_text, no_think_text)
                    # 额外保存 uuid 和 domain 供 Step 3 过滤使用
                    no_think_item = {
                        "uuid": uuid,
                        "domain": domain,
                        **no_think_sharegpt,
                    }
                    f_inst.write(json.dumps(no_think_item, ensure_ascii=False) + "\n")
                    instruction_no_think_count += 1
                    domain_stats[domain]["no_think_extracted"] += 1
                else:
                    instruction_extract_fail_count += 1
            else:
                # math + science: 提取 ground truth，写入 metadata
                gt = extract_ground_truth(output_text, domain)
                if gt is not None:
                    gt_found_count += 1
                    domain_stats[domain]["gt_found"] = domain_stats[domain].get("gt_found", 0) + 1
                else:
                    gt_missing_count += 1

                meta = {
                    "uuid": uuid,
                    "input": input_text,
                    "domain": domain,
                    "ground_truth": gt,
                    "stage": stage,
                }
                f_meta.write(json.dumps(meta, ensure_ascii=False) + "\n")
                metadata_count += 1

            # 进度日志
            if think_count % 10000 == 0:
                print(f"  已处理: {think_count} 条...")

    # ---- 4. 输出统计 ----
    print(f"\n{'─'*40}")
    print("4. 统计结果")
    print(f"  /think 样本总数: {think_count}")
    print(f"  metadata 样本数 (math+science): {metadata_count}")
    print(f"  Ground truth 提取成功: {gt_found_count} ({gt_found_count/max(metadata_count,1)*100:.1f}%)")
    print(f"  Ground truth 提取失败: {gt_missing_count} ({gt_missing_count/max(metadata_count,1)*100:.1f}%)")
    print(f"  instruction /no_think 提取成功: {instruction_no_think_count}")
    print(f"  instruction /no_think 提取失败: {instruction_extract_fail_count}")
    print(f"\n  各 domain 统计:")
    for domain, stats in sorted(domain_stats.items()):
        total = stats["total"]
        if domain == "instruction-following":
            extracted = stats.get("no_think_extracted", 0)
            print(f"    {domain}: {total} 条, /no_think 提取率 {extracted/max(total,1)*100:.1f}%")
        else:
            found = stats.get("gt_found", 0)
            print(f"    {domain}: {total} 条, GT 提取率 {found/max(total,1)*100:.1f}%")

    print(f"\n  输出文件:")
    print(f"    /think 数据:              {think_output}")
    print(f"    元数据 (math+science):    {metadata_output}")
    print(f"    /no_think (instruction):  {instruction_no_think_output}")
    print(f"\n{'='*60}")
    print("Step 1 完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
