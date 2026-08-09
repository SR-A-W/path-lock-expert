#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提取失败样本的 UUID

两种模式：
1. 默认模式：从 *_failed.json 文件中提取所有失败样本的 UUID
2. 对比模式：对比原始数据集（metadata.jsonl）和生成的数据集（no_think_raw.jsonl），
   找出未成功生成的样本

使用场景：
  当两个实验误用同一个 rejected_uuids.jsonl 时，可以通过此脚本
  重新生成正确的失败样本列表。

使用方法:
    conda activate het

    # 默认模式：从 *_failed.json 提取
    python -m src.datasets.extract_failed_uuids \
        --failed_json datasets/superior_reasoning/no_think_raw_failed.json

    # 指定输出文件名
    python -m src.datasets.extract_failed_uuids \
        --failed_json datasets/superior_reasoning/no_think_raw_failed.json \
        --output datasets/superior_reasoning/rejected_uuids.jsonl

    # 对比模式：对比原始数据集和生成数据集
    python -m src.datasets.extract_failed_uuids \
        --compare \
        --metadata datasets/superior_reasoning/metadata.jsonl \
        --generated datasets/superior_reasoning/no_think_raw.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# 项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_uuids_from_jsonl(file_path: Path) -> set:
    """从 JSONL 文件中加载所有 UUID

    Args:
        file_path: JSONL 文件路径

    Returns:
        UUID 集合
    """
    uuids = set()
    if not file_path.exists():
        print(f"警告: 文件不存在: {file_path}")
        return uuids

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if "uuid" in item:
                    uuids.add(item["uuid"])
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行 JSON 解析失败: {e}")
                continue

    return uuids


def load_metadata_with_domain(file_path: Path) -> dict:
    """从 metadata.jsonl 加载 UUID 及其对应的 domain

    Args:
        file_path: metadata.jsonl 文件路径

    Returns:
        {uuid: {"domain": domain, "input": input_text}} 字典
    """
    uuid_info = {}
    if not file_path.exists():
        print(f"错误: 文件不存在: {file_path}")
        return uuid_info

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                uuid = item.get("uuid")
                if uuid:
                    uuid_info[uuid] = {
                        "domain": item.get("domain", "unknown"),
                        "input": item.get("input", ""),
                    }
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行 JSON 解析失败: {e}")
                continue

    return uuid_info


def load_failed_samples_from_json(file_path: Path) -> list:
    """从 *_failed.json 文件中加载失败样本

    Args:
        file_path: *_failed.json 文件路径

    Returns:
        失败样本列表，每个元素包含 uuid, domain, reason, input 等字段
    """
    if not file_path.exists():
        print(f"错误: 文件不存在: {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            failed_samples = data.get("failed_samples", [])
            return failed_samples
    except json.JSONDecodeError as e:
        print(f"错误: JSON 解析失败: {e}")
        return []
    except Exception as e:
        print(f"错误: 读取文件失败: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="提取失败样本的 UUID（默认从 *_failed.json，或对比原始数据集和生成数据集）"
    )

    # 模式选择
    parser.add_argument(
        "--compare",
        action="store_true",
        help="使用对比模式（对比 metadata 和 generated 文件）",
    )

    # 默认模式参数
    parser.add_argument(
        "--failed_json",
        type=str,
        default=None,
        help="失败样本 JSON 文件路径（*_failed.json）",
    )

    # 对比模式参数
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="原始数据集路径（metadata.jsonl，仅对比模式需要）",
    )
    parser.add_argument(
        "--generated",
        type=str,
        default=None,
        help="生成的数据集路径（no_think_raw.jsonl，仅对比模式需要）",
    )

    # 通用参数
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（默认：根据输入文件自动生成）",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "jsonl"],
        default="jsonl",
        help="输出格式：json（单个 JSON 数组）或 jsonl（每行一个 JSON 对象，默认）",
    )
    args = parser.parse_args()

    # 参数验证
    if args.compare:
        # 对比模式
        if not args.metadata or not args.generated:
            print("错误: 对比模式需要同时指定 --metadata 和 --generated")
            return

        metadata_path = project_root / args.metadata
        generated_path = project_root / args.generated

        # 输出路径：默认为生成文件名 + _rejected_uuid.jsonl
        if args.output:
            output_path = project_root / args.output
        else:
            suffix = ".json" if args.format == "json" else ".jsonl"
            output_path = generated_path.parent / (generated_path.stem + "_rejected_uuid" + suffix)

        run_compare_mode(metadata_path, generated_path, output_path, args.format)
    else:
        # 默认模式：从 *_failed.json 提取
        if not args.failed_json:
            print("错误: 默认模式需要指定 --failed_json")
            print("提示: 使用 --compare 切换到对比模式")
            return

        failed_json_path = project_root / args.failed_json

        # 输出路径：默认为同目录下的 rejected_uuids.jsonl
        if args.output:
            output_path = project_root / args.output
        else:
            output_path = failed_json_path.parent / "rejected_uuids.jsonl"

        run_default_mode(failed_json_path, output_path, args.format)


def run_default_mode(failed_json_path: Path, output_path: Path, output_format: str):
    """默认模式：从 *_failed.json 提取失败样本 UUID

    Args:
        failed_json_path: *_failed.json 文件路径
        output_path: 输出文件路径
        output_format: 输出格式（json 或 jsonl）
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("提取失败样本 UUID（默认模式）")
    print(f"{'='*60}")
    print(f"失败样本文件: {failed_json_path}")
    print(f"输出文件:     {output_path}")
    print(f"输出格式:     {output_format}")

    # ---- 1. 加载失败样本 ----
    print(f"\n{'─'*40}")
    print("1. 加载失败样本...")

    failed_samples = load_failed_samples_from_json(failed_json_path)
    print(f"  失败样本数: {len(failed_samples)}")

    if not failed_samples:
        print("\n  未找到失败样本。")
        return

    # ---- 2. 按 domain 和 error_type 统计 ----
    print(f"\n{'─'*40}")
    print("2. 统计失败样本...")

    domain_stats = {}
    error_type_stats = {}

    for sample in failed_samples:
        domain = sample.get("domain", "unknown")
        error_type = sample.get("error_type", "unknown")

        domain_stats[domain] = domain_stats.get(domain, 0) + 1
        error_type_stats[error_type] = error_type_stats.get(error_type, 0) + 1

    print(f"\n  失败样本按 domain 分布:")
    for domain, count in sorted(domain_stats.items()):
        print(f"    {domain}: {count}")

    print(f"\n  失败样本按 error_type 分布:")
    for error_type, count in sorted(error_type_stats.items()):
        print(f"    {error_type}: {count}")

    # ---- 3. 准备输出数据 ----
    print(f"\n{'─'*40}")
    print("3. 准备输出数据...")

    output_items = []
    for sample in failed_samples:
        output_items.append({
            "uuid": sample.get("uuid", ""),
            "domain": sample.get("domain", "unknown"),
            "reason": f"generation_failed_{sample.get('error_type', 'unknown')}",
            "input": sample.get("input", ""),
        })

    # ---- 4. 写入输出文件 ----
    print(f"\n{'─'*40}")
    print("4. 写入输出文件...")

    if output_format == "json":
        # JSON 格式：单个数组
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_items, f, ensure_ascii=False, indent=2)
    else:
        # JSONL 格式：每行一个对象
        with open(output_path, "w", encoding="utf-8") as f:
            for item in output_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  已写入 {len(output_items)} 条失败样本")
    print(f"  输出文件: {output_path}")

    print(f"\n{'='*60}")
    print("完成!")
    print(f"{'='*60}")


def run_compare_mode(metadata_path: Path, generated_path: Path, output_path: Path, output_format: str):
    """对比模式：对比原始数据集和生成数据集，找出失败样本

    Args:
        metadata_path: metadata.jsonl 文件路径
        generated_path: 生成的数据集文件路径
        output_path: 输出文件路径
        output_format: 输出格式（json 或 jsonl）
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("提取失败样本 UUID（对比模式）")
    print(f"{'='*60}")
    print(f"原始数据集: {metadata_path}")
    print(f"生成数据集: {generated_path}")
    print(f"输出文件:   {output_path}")
    print(f"输出格式:   {output_format}")

    # ---- 1. 加载原始数据集的 UUID 和 domain ----
    print(f"\n{'─'*40}")
    print("1. 加载原始数据集...")

    metadata_info = load_metadata_with_domain(metadata_path)
    print(f"  原始数据集 UUID 数: {len(metadata_info)}")

    if not metadata_info:
        print("错误: 原始数据集为空或无法读取")
        return

    # ---- 2. 加载生成数据集的 UUID ----
    print(f"\n{'─'*40}")
    print("2. 加载生成数据集...")

    generated_uuids = load_uuids_from_jsonl(generated_path)
    print(f"  生成数据集 UUID 数: {len(generated_uuids)}")

    # ---- 3. 找出失败的 UUID ----
    print(f"\n{'─'*40}")
    print("3. 对比并提取失败样本...")

    failed_uuids = set(metadata_info.keys()) - generated_uuids
    print(f"  失败样本数: {len(failed_uuids)}")

    if not failed_uuids:
        print("\n  所有样本都已成功生成，无失败样本。")
        return

    # ---- 4. 按 domain 统计 ----
    domain_stats = {}
    failed_items = []

    for uuid in failed_uuids:
        info = metadata_info[uuid]
        domain = info["domain"]
        domain_stats[domain] = domain_stats.get(domain, 0) + 1

        failed_items.append({
            "uuid": uuid,
            "domain": domain,
            "reason": "not_generated",
            "input": info["input"],
        })

    print(f"\n  失败样本按 domain 分布:")
    for domain, count in sorted(domain_stats.items()):
        print(f"    {domain}: {count}")

    # ---- 5. 写入输出文件 ----
    print(f"\n{'─'*40}")
    print("4. 写入输出文件...")

    if output_format == "json":
        # JSON 格式：单个数组
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(failed_items, f, ensure_ascii=False, indent=2)
    else:
        # JSONL 格式：每行一个对象
        with open(output_path, "w", encoding="utf-8") as f:
            for item in failed_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  已写入 {len(failed_items)} 条失败样本")
    print(f"  输出文件: {output_path}")

    print(f"\n{'='*60}")
    print("完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
