#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 /think 和 /no_think 注册为 PLE tokenizer 的 special token。

功能：
1. 在 tokenizer 中添加 /think 和 /no_think 作为 additional_special_tokens
   - 它们将作为整体 token 被识别，不会被 BPE 拆分
   - 分配新的 token ID（在现有 special token 之后）
2. 更新 config.json 中的 think_token_id 和 no_think_token_id
3. 若新 token ID 超出原始 vocab_size（如 LLaMA 3.1），自动 resize embedding 和 lm_head
4. 验证注册结果

使用方式：
    python -m src.utils.register_control_tokens --model_path models/llama_ple_initialized

注意事项：
- Qwen 系列: vocab_size 有预留空间（152064），新 token 不需要 resize
- LLaMA 3.1: vocab_size=128256 全部用满，新 token ID 128256/128257 需要 resize embedding
- resize 不破坏原始权重：前 N 行完全保留，仅在末尾追加随机初始化的新行
- 新 token 的 embedding 权重为未训练状态（随机初始化值），fine-tune 时会学习
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# 默认路径（向后兼容）
DEFAULT_PLE_PATH = os.path.join(PROJECT_ROOT, "models", "ple_initialized")

# 要注册的控制 token
THINK_TOKEN = "/think"
NO_THINK_TOKEN = "/no_think"


def main():
    parser = argparse.ArgumentParser(
        description="将 /think 和 /no_think 注册为 PLE tokenizer 的 special token"
    )
    parser.add_argument(
        "--model_path", type=str, default=DEFAULT_PLE_PATH,
        help=f"PLE 模型路径 (默认: {DEFAULT_PLE_PATH})"
    )
    args = parser.parse_args()
    PLE_PATH = args.model_path

    from transformers import AutoTokenizer

    print(f"模型路径: {PLE_PATH}")

    # ===== 1. 加载 tokenizer =====
    print("\n[1] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(PLE_PATH, trust_remote_code=True)
    print(f"  当前 vocab 大小: {len(tokenizer)}")
    print(f"  当前 special tokens 数量: {len(tokenizer.all_special_tokens)}")

    # 检查是否已注册
    existing_special = tokenizer.special_tokens_map.get("additional_special_tokens", [])
    # additional_special_tokens 可能是 AddedToken 对象列表，需要转为字符串
    existing_str = [str(t) for t in existing_special]
    if THINK_TOKEN in existing_str and NO_THINK_TOKEN in existing_str:
        think_id = tokenizer.convert_tokens_to_ids(THINK_TOKEN)
        no_think_id = tokenizer.convert_tokens_to_ids(NO_THINK_TOKEN)
        print(f"\n  控制 token 已注册：")
        print(f"    {THINK_TOKEN} → id={think_id}")
        print(f"    {NO_THINK_TOKEN} → id={no_think_id}")
        print("  跳过注册步骤，直接验证...")
    else:
        # ===== 2. 注册新的 special token =====
        print(f"\n[2] 注册控制 token: {THINK_TOKEN}, {NO_THINK_TOKEN}")

        # 构建新的 additional_special_tokens 列表（保留现有的）
        new_special_tokens = existing_str + [THINK_TOKEN, NO_THINK_TOKEN]
        num_added = tokenizer.add_special_tokens({
            "additional_special_tokens": new_special_tokens
        })
        print(f"  新增 token 数量: {num_added}")

        think_id = tokenizer.convert_tokens_to_ids(THINK_TOKEN)
        no_think_id = tokenizer.convert_tokens_to_ids(NO_THINK_TOKEN)
        print(f"  {THINK_TOKEN} → id={think_id}")
        print(f"  {NO_THINK_TOKEN} → id={no_think_id}")

        # ===== 3. 保存更新后的 tokenizer =====
        print(f"\n[3] 保存 tokenizer 到 {PLE_PATH}")
        tokenizer.save_pretrained(PLE_PATH)
        print("  tokenizer 已保存")

    # ===== 4. 更新 config.json =====
    config_path = os.path.join(PLE_PATH, "config.json")
    print(f"\n[4] 更新 config.json...")
    with open(config_path, "r") as f:
        config = json.load(f)

    old_think = config.get("think_token_id")
    old_no_think = config.get("no_think_token_id")
    config["think_token_id"] = think_id
    config["no_think_token_id"] = no_think_id

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")  # 文件末尾换行

    print(f"  think_token_id:    {old_think} → {think_id}")
    print(f"  no_think_token_id: {old_no_think} → {no_think_id}")

    # ===== 4.5. Resize embedding（仅在 token ID 超出 vocab_size 时） =====
    max_new_id = max(think_id, no_think_id)
    current_vocab_size = config.get("vocab_size", 0)
    if max_new_id >= current_vocab_size:
        new_vocab_size = max_new_id + 1
        print(f"\n[4.5] Resize embedding: {current_vocab_size} → {new_vocab_size}")
        print(f"  新 token ID ({max_new_id}) >= 原始 vocab_size ({current_vocab_size})，需要 resize")
        print(f"  原始 {current_vocab_size} 行 embedding 完全保留，追加 {new_vocab_size - current_vocab_size} 行随机初始化")

        from transformers import AutoModelForCausalLM
        import torch

        print(f"  加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            PLE_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model.resize_token_embeddings(new_vocab_size)
        print(f"  embed_tokens shape: {model.get_input_embeddings().weight.shape}")
        if hasattr(model, 'lm_head'):
            print(f"  lm_head shape: {model.lm_head.weight.shape}")

        print(f"  保存 resize 后的模型权重...")
        model.save_pretrained(PLE_PATH)
        print(f"  模型权重已保存")

        # 更新 config.json 中的 vocab_size
        with open(config_path, "r") as f:
            config = json.load(f)
        config["vocab_size"] = new_vocab_size
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"  config.json vocab_size 已更新: {current_vocab_size} → {new_vocab_size}")

        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"\n  vocab_size={current_vocab_size} 足够容纳新 token (max_id={max_new_id})，无需 resize")

    # ===== 5. 验证 =====
    print(f"\n[5] 验证...")

    # 重新加载验证
    tokenizer2 = AutoTokenizer.from_pretrained(PLE_PATH, trust_remote_code=True)

    # 验证单 token 编码
    think_encoded = tokenizer2.encode(THINK_TOKEN, add_special_tokens=False)
    no_think_encoded = tokenizer2.encode(NO_THINK_TOKEN, add_special_tokens=False)

    assert len(think_encoded) == 1, (
        f"{THINK_TOKEN} 应编码为单个 token，实际: {think_encoded}"
    )
    assert len(no_think_encoded) == 1, (
        f"{NO_THINK_TOKEN} 应编码为单个 token，实际: {no_think_encoded}"
    )
    assert think_encoded[0] == think_id, (
        f"{THINK_TOKEN} 编码 ID 不匹配: {think_encoded[0]} != {think_id}"
    )
    assert no_think_encoded[0] == no_think_id, (
        f"{NO_THINK_TOKEN} 编码 ID 不匹配: {no_think_encoded[0]} != {no_think_id}"
    )

    # 验证在句子中不会被拆分
    test_text = f"Hello {THINK_TOKEN} world"
    test_ids = tokenizer2.encode(test_text, add_special_tokens=False)
    assert think_id in test_ids, (
        f"{THINK_TOKEN} 在句子中未被正确识别为单 token: {test_ids}"
    )

    test_text2 = f"Hello {NO_THINK_TOKEN} world"
    test_ids2 = tokenizer2.encode(test_text2, add_special_tokens=False)
    assert no_think_id in test_ids2, (
        f"{NO_THINK_TOKEN} 在句子中未被正确识别为单 token: {test_ids2}"
    )

    # 验证 decode 正确
    decoded_think = tokenizer2.decode([think_id])
    decoded_no_think = tokenizer2.decode([no_think_id])
    assert decoded_think.strip() == THINK_TOKEN, (
        f"decode 不一致: {repr(decoded_think)} != {repr(THINK_TOKEN)}"
    )
    assert decoded_no_think.strip() == NO_THINK_TOKEN, (
        f"decode 不一致: {repr(decoded_no_think)} != {repr(NO_THINK_TOKEN)}"
    )

    # 验证 config 已更新
    with open(config_path, "r") as f:
        config_check = json.load(f)
    assert config_check["think_token_id"] == think_id
    assert config_check["no_think_token_id"] == no_think_id

    # 验证 embedding 层大小足够（resize 后应当满足）
    vocab_size = config_check.get("vocab_size", 0)
    assert vocab_size > max(think_id, no_think_id), (
        f"vocab_size={vocab_size} 不足以容纳新 token (max_id={max(think_id, no_think_id)})"
    )

    print(f"\n  所有验证通过！")
    print(f"\n{'='*60}")
    print(f"  注册完成总结：")
    print(f"    {THINK_TOKEN:12s} → id={think_id}")
    print(f"    {NO_THINK_TOKEN:12s} → id={no_think_id}")
    print(f"    vocab_size={vocab_size}")
    print(f"    config.json 已更新")
    print(f"    tokenizer 已更新")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
