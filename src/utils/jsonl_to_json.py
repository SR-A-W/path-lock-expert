"""
JSONL to JSON Converter

用法:
  # 顺序抽取前100条
  python jsonl_to_json.py data.jsonl 100

  # 随机乱序抽取100条
  python jsonl_to_json.py data.jsonl 100 --shuffle
"""

import json
import argparse
import random
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Convert JSONL to pretty JSON file.")
    parser.add_argument("input", type=str, help="目标 .jsonl 文件路径")
    parser.add_argument("count", type=int, help="要转换的数据条数")
    parser.add_argument("--shuffle", action="store_true", default=False,
                        help="是否随机乱序抽取（默认按顺序抽取最前的数据）")
    return parser.parse_args()


def main():
    args = parse_args()
    # 检查文件存在
    if not os.path.isfile(args.input):
        print(f"[Error] 文件不存在: {args.input}")
        sys.exit(1)

    # 读取所有行
    with open(args.input, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    total = len(lines)
    print(f"[Info] 文件共 {total} 条数据")

    if args.count > total:
        print(f"[Warning] 请求 {args.count} 条，但文件只有 {total} 条，将全部导出")
        args.count = total

    # 按需抽取
    if args.shuffle:
        selected_lines = random.sample(lines, args.count)
        print(f"[Info] 随机抽取 {args.count} 条")
    else:
        selected_lines = lines[:args.count]
        print(f"[Info] 顺序抽取前 {args.count} 条")

    # 解析 JSON
    records = []
    for i, line in enumerate(selected_lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[Warning] 第 {i+1} 条解析失败，已跳过: {e}")

    # 生成输出路径（同名同路径，后缀改为 .json）
    base, _ = os.path.splitext(args.input)
    output_path = base + ".json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[Done] 已保存 {len(records)} 条到: {output_path}")


if __name__ == "__main__":
    main()