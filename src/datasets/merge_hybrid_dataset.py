#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并 /think 和 /no_think 数据为最终训练集

读取 prepare_think_data.py 产出的 think_data.jsonl 和
filter_no_think_data.py 产出的 no_think_filtered.jsonl，
按指定比例采样，输出最终训练用 JSONL。

不做随机打乱（LLaMA-Factory 的 RandomSampler 会自动处理）。

支持一次生成多个比例变体（如 1:1, 1:2, 2:1）。

输出到 datasets/superior_reasoning/final/，用户手动复制到 train/data/ 并注册。

使用方法:
    conda activate het
    python -m src.datasets.merge_hybrid_dataset \
        --think_input datasets/superior_reasoning/think_data.jsonl \
        --no_think_input datasets/superior_reasoning/no_think_filtered.jsonl \
        --output_dir datasets/superior_reasoning/final \
        --ratios 1:1,1:2,2:1

依赖:
    无额外依赖（仅标准库）
"""

import argparse
import json
import random
import sys
from pathlib import Path

# 项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ==================== 工具函数 ====================

def load_jsonl(path: str) -> list:
    """加载 JSONL 文件"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(data: list, path: str):
    """写入 JSONL 文件"""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_ratio(ratio_str: str) -> tuple[int, int]:
    """解析比例字符串（如 '1:2' → (1, 2)）"""
    parts = ratio_str.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"无效比例格式: {ratio_str}，应为 'X:Y'")
    return int(parts[0]), int(parts[1])


def compute_counts(
    n_think: int, n_no_think: int, ratio_think: int, ratio_no_think: int
) -> tuple[int, int]:
    """根据比例和可用数据量计算实际采样数

    以数据量较少的一方为瓶颈，按比例缩放另一方。

    Args:
        n_think: 可用 think 样本数
        n_no_think: 可用 no_think 样本数
        ratio_think: think 比例系数
        ratio_no_think: no_think 比例系数

    Returns:
        (think_count, no_think_count) 实际采样数
    """
    per_unit_think = n_think / ratio_think
    per_unit_no_think = n_no_think / ratio_no_think
    per_unit = min(per_unit_think, per_unit_no_think)

    think_count = int(per_unit * ratio_think)
    no_think_count = int(per_unit * ratio_no_think)

    return think_count, no_think_count


# dataset_info.json 注册模板
DATASET_INFO_TEMPLATE = {
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "system": None},
    "tags": {
        "role_tag": "from",
        "content_tag": "value",
        "user_tag": "user",
        "assistant_tag": "assistant",
    },
}


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(
        description="合并 /think 和 /no_think 数据为最终训练集"
    )
    parser.add_argument(
        "--think_input", type=str,
        default="datasets/superior_reasoning/think_data.jsonl",
        help="think 数据路径",
    )
    parser.add_argument(
        "--no_think_input", type=str,
        default="datasets/superior_reasoning/no_think_filtered.jsonl",
        help="no_think 数据路径",
    )
    parser.add_argument(
        "--output_dir", type=str, default="datasets/superior_reasoning/final",
        help="输出目录",
    )
    parser.add_argument(
        "--output_name", type=str, default="superior_reasoning",
        help="输出文件名前缀",
    )
    parser.add_argument(
        "--ratios", type=str, default="1:1",
        help="think:no_think 比例，逗号分隔多个（如 1:1,1:2,2:1）",
    )
    parser.add_argument(
        "--max_total", type=int, default=0,
        help="最大总样本数（0=不限制）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子（用于采样）")
    args = parser.parse_args()

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ratios = [parse_ratio(r) for r in args.ratios.split(",")]

    print(f"\n{'='*60}")
    print("Step 4: 合并最终训练集")
    print(f"{'='*60}")
    print(f"think 数据:    {args.think_input}")
    print(f"no_think 数据: {args.no_think_input}")
    print(f"比例变体:      {args.ratios}")
    print(f"随机种子:      {args.seed}")
    print(f"注意: 不做随机打乱（LLaMA-Factory RandomSampler 自动处理）")

    # ---- 1. 加载数据 ----
    print(f"\n{'─'*40}")
    print("1. 加载数据...")
    think_data = load_jsonl(str(project_root / args.think_input))
    no_think_data = load_jsonl(str(project_root / args.no_think_input))
    print(f"  think 样本数:    {len(think_data)}")
    print(f"  no_think 样本数: {len(no_think_data)}")

    # ---- 2. 按比例生成各变体 ----
    print(f"\n{'─'*40}")
    print("2. 生成比例变体...")

    registry_entries = {}

    for rt, rn in ratios:
        ratio_label = f"{rt}_{rn}"
        tc, nc = compute_counts(len(think_data), len(no_think_data), rt, rn)

        # 可选：限制总量
        if args.max_total > 0 and (tc + nc) > args.max_total:
            scale = args.max_total / (tc + nc)
            tc = int(tc * scale)
            nc = int(nc * scale)

        random.seed(args.seed)
        sampled_think = random.sample(think_data, tc)
        sampled_no_think = random.sample(no_think_data, nc)

        # 不做 shuffle，直接拼接（think 在前，no_think 在后）
        combined = sampled_think + sampled_no_think

        filename = f"{args.output_name}_{ratio_label}.jsonl"
        output_path = output_dir / filename
        write_jsonl(combined, str(output_path))

        print(f"\n  比例 {rt}:{rn}:")
        print(f"    think: {tc}, no_think: {nc}, 总计: {tc + nc}")
        print(f"    输出: {output_path}")

        # 注册条目
        entry_name = f"{args.output_name}_{ratio_label}"
        registry_entries[entry_name] = {
            "file_name": filename,
            **DATASET_INFO_TEMPLATE,
        }

    # ---- 3. 打印注册条目 ----
    print(f"\n{'─'*40}")
    print("3. dataset_info.json 注册条目（请手动添加到 train/data/dataset_info.json）:")
    print(json.dumps(registry_entries, indent=2, ensure_ascii=False))

    print(f"\n{'─'*40}")
    print("提示: 请手动将需要的成品文件从")
    print(f"  {output_dir}/")
    print("复制到 train/data/ 并在 dataset_info.json 中注册。")

    print(f"\n{'='*60}")
    print("Step 4 完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
