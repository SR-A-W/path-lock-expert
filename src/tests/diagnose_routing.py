#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PLE 路由诊断脚本（4-bit 量化，全部加载到 GPU）

诊断目标：
1. 验证 /think 和 /no_think 控制 token 的注册和编码
2. 验证 _determine_routing_index 能从 input_ids 中正确检测控制 token
3. 验证 generate 中包含控制 token 时路由自动切换
4. 验证 routing_index 在 forward 和 generate 中是否正确传递
5. 确认 expert 0 / expert 1 的输出差异

运行方式:
    cd ./
    conda activate het
    python src/tests/diagnose_routing.py
"""

import gc
import sys
import os
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLE_PATH = os.path.join(PROJECT_ROOT, "models", "ple_initialized")
QWEN_PATH = os.path.join(PROJECT_ROOT, "models", "source", "Qwen2.5-7B-Instruct")
DEEPSEEK_PATH = os.path.join(PROJECT_ROOT, "models", "source", "DeepSeek-R1-Distill-Qwen-7B")


def load_model_4bit(model_path, trust_remote_code=False):
    """以 4-bit 量化加载模型到 GPU。"""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model


def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path)


def generate_text(model, tokenizer, prompt, max_new_tokens=80, routing_index=None):
    """生成文本，可选指定 routing_index。"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    if routing_index is not None:
        gen_kwargs["routing_index"] = routing_index

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # 只取新生成的部分
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    # ===== 诊断 1: 控制 token 注册验证 =====
    section("诊断 1: 控制 token 注册验证")

    tokenizer = load_tokenizer(PLE_PATH)

    # 检查 /think 和 /no_think 是否为 special token
    print(f"  tokenizer 中的 additional_special_tokens:")
    for t in tokenizer.special_tokens_map.get("additional_special_tokens", []):
        print(f"    {repr(str(t))}")

    # 编码验证：必须是单 token
    think_ids = tokenizer.encode("/think", add_special_tokens=False)
    no_think_ids = tokenizer.encode("/no_think", add_special_tokens=False)
    print(f"\n  /think    encode → {think_ids} (单token={'✓' if len(think_ids)==1 else '✗'})")
    print(f"  /no_think encode → {no_think_ids} (单token={'✓' if len(no_think_ids)==1 else '✗'})")

    think_token_id = think_ids[0] if len(think_ids) == 1 else None
    no_think_token_id = no_think_ids[0] if len(no_think_ids) == 1 else None
    print(f"\n  think_token_id    = {think_token_id}")
    print(f"  no_think_token_id = {no_think_token_id}")

    # 在句子中验证不被拆分
    test_text = "Hello /think world /no_think end"
    test_enc = tokenizer.encode(test_text, add_special_tokens=False)
    print(f"\n  句子: {repr(test_text)}")
    print(f"  编码: {test_enc}")
    print(f"  /think  在编码中: {'✓' if think_token_id in test_enc else '✗'}")
    print(f"  /no_think 在编码中: {'✓' if no_think_token_id in test_enc else '✗'}")

    # ===== 诊断 2: _determine_routing_index 检测控制 token =====
    section("诊断 2: _determine_routing_index 控制 token 检测")

    # 内联路由函数（与 modeling_qwen2_ple.py 一致）
    class FakeConfig:
        pass

    def _determine_routing_index_test(input_ids, config):
        think_id = config.think_token_id
        no_think_id = config.no_think_token_id
        default_idx = getattr(config, "default_routing_index", 0)
        if think_id is None and no_think_id is None:
            return default_idx
        last_think_pos = -1
        last_no_think_pos = -1
        flat = input_ids.view(-1)
        if think_id is not None:
            positions = (flat == think_id).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                last_think_pos = positions[-1].item()
        if no_think_id is not None:
            positions = (flat == no_think_id).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                last_no_think_pos = positions[-1].item()
        if last_think_pos == -1 and last_no_think_pos == -1:
            return default_idx
        if last_think_pos > last_no_think_pos:
            return 1
        else:
            return 0

    cfg = FakeConfig()
    cfg.think_token_id = think_token_id
    cfg.no_think_token_id = no_think_token_id
    cfg.default_routing_index = 0

    # 测试各种场景
    cases = [
        ("无控制 token",       "What is 2+3?"),
        ("含 /think",          "What is 2+3? /think"),
        ("含 /no_think",       "What is 2+3? /no_think"),
        ("/think 在 /no_think 之后", "What /no_think is /think 2+3?"),
        ("/no_think 在 /think 之后", "What /think is /no_think 2+3?"),
    ]

    for name, text in cases:
        ids = torch.tensor([tokenizer.encode(text, add_special_tokens=False)])
        result = _determine_routing_index_test(ids, cfg)
        print(f"  {name:30s} → routing_index = {result}")

    # ===== 诊断 3: 加载 PLE 模型并测试 forward 路由 =====
    section("诊断 3: PLE forward 路由测试（4-bit）")

    print("  加载 PLE 模型（4-bit）...")
    pl_model = load_model_4bit(PLE_PATH, trust_remote_code=True)

    # 检查 config 中的路由配置
    print(f"  config.think_token_id     = {pl_model.config.think_token_id}")
    print(f"  config.no_think_token_id  = {pl_model.config.no_think_token_id}")
    print(f"  config.default_routing_index = {pl_model.config.default_routing_index}")

    # 测试 forward 的 routing_index 传递
    input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(pl_model.device)
    with torch.no_grad():
        logits_r0 = pl_model(input_ids=input_ids, routing_index=0).logits.cpu()
        logits_r1 = pl_model(input_ids=input_ids, routing_index=1).logits.cpu()
        # 不传 routing_index → 走 _determine_routing_index → 返回 default
        logits_default = pl_model(input_ids=input_ids).logits.cpu()

    print(f"\n  forward 结果:")
    print(f"  routing_index=0  top5 token ids: {logits_r0[0, -1].topk(5).indices.tolist()}")
    print(f"  routing_index=1  top5 token ids: {logits_r1[0, -1].topk(5).indices.tolist()}")
    print(f"  无 routing_index top5 token ids: {logits_default[0, -1].topk(5).indices.tolist()}")
    print(f"  routing_index=0 == default?  {torch.equal(logits_r0, logits_default)}")
    print(f"  routing_index=1 == default?  {torch.equal(logits_r1, logits_default)}")
    print(f"  routing_index=0 == routing_index=1?  {torch.equal(logits_r0, logits_r1)}")

    # ===== 诊断 4: generate 显式路由测试 =====
    section("诊断 4: PLE generate 显式路由测试")

    prompt = "What is 2 + 3? Answer:"

    print(f"\n  Prompt: '{prompt}'")

    try:
        text_r0 = generate_text(pl_model, tokenizer, prompt, routing_index=0)
        print(f"\n  [generate routing_index=0] {text_r0[:200]}")
    except Exception as e:
        print(f"\n  [generate routing_index=0] ERROR: {e}")

    try:
        text_r1 = generate_text(pl_model, tokenizer, prompt, routing_index=1)
        print(f"\n  [generate routing_index=1] {text_r1[:200]}")
    except Exception as e:
        print(f"\n  [generate routing_index=1] ERROR: {e}")

    try:
        text_default = generate_text(pl_model, tokenizer, prompt)
        print(f"\n  [generate 无 routing_index] {text_default[:200]}")
    except Exception as e:
        print(f"\n  [generate 无 routing_index] ERROR: {e}")

    if text_r0 == text_r1:
        print("\n  ✗ BUG: routing_index=0 和 =1 的 generate 输出完全相同！路由未生效！")
    else:
        print("\n  ✓ routing_index=0 和 =1 的 generate 输出不同，显式路由已生效。")

    print(f"  最后缓存的 _cached_routing_index = {pl_model._cached_routing_index}")

    # ===== 诊断 5: generate 控制 token 自动路由测试（核心新增） =====
    section("诊断 5: PLE generate 控制 token 自动路由测试")

    prompt_think = "What is 2 + 3? /think Answer:"
    prompt_no_think = "What is 2 + 3? /no_think Answer:"
    prompt_plain = "What is 2 + 3? Answer:"

    print(f"\n  含 /think 的 prompt:    '{prompt_think}'")
    print(f"  含 /no_think 的 prompt: '{prompt_no_think}'")
    print(f"  无控制 token 的 prompt: '{prompt_plain}'")

    try:
        text_think = generate_text(pl_model, tokenizer, prompt_think)
        cached_think = pl_model._cached_routing_index
        print(f"\n  [/think]    routing={cached_think} → {text_think[:200]}")
    except Exception as e:
        print(f"\n  [/think]    ERROR: {e}")
        cached_think = None

    try:
        text_no_think = generate_text(pl_model, tokenizer, prompt_no_think)
        cached_no_think = pl_model._cached_routing_index
        print(f"\n  [/no_think] routing={cached_no_think} → {text_no_think[:200]}")
    except Exception as e:
        print(f"\n  [/no_think] ERROR: {e}")
        cached_no_think = None

    try:
        text_plain = generate_text(pl_model, tokenizer, prompt_plain)
        cached_plain = pl_model._cached_routing_index
        print(f"\n  [无控制]    routing={cached_plain} → {text_plain[:200]}")
    except Exception as e:
        print(f"\n  [无控制]    ERROR: {e}")
        cached_plain = None

    # 验证路由检测结果
    print(f"\n  路由检测结果:")
    print(f"    /think    → _cached_routing_index = {cached_think} (期望 1)")
    print(f"    /no_think → _cached_routing_index = {cached_no_think} (期望 0)")
    print(f"    无控制    → _cached_routing_index = {cached_plain} (期望 0, default)")

    if cached_think == 1 and cached_no_think == 0 and cached_plain == 0:
        print("\n  ✓ 控制 token 自动路由完全正确！")
    else:
        print("\n  ✗ 控制 token 自动路由存在问题！")

    if text_think != text_no_think:
        print("  ✓ /think 和 /no_think 产生不同输出")
    else:
        print("  ✗ /think 和 /no_think 输出相同")

    if text_no_think == text_plain:
        print("  ✓ /no_think 和无控制 token 输出相同（都走 default=0）")
    else:
        print("  ~ /no_think 和无控制 token 输出不同（prompt 不同导致，可接受）")

    # 释放 PLE 模型
    del pl_model
    gc.collect()
    torch.cuda.empty_cache()

    # ===== 总结 =====
    section("诊断总结")
    print("""
  1. /think 和 /no_think 注册为单 token special token？(见诊断 1)
  2. _determine_routing_index 能正确检测控制 token？(见诊断 2)
  3. forward 中 routing_index 正确传递？(见诊断 3)
  4. generate 中显式 routing_index 正确传递？(见诊断 4)
  5. generate 中控制 token 自动触发正确路由？(见诊断 5)
    """)


if __name__ == "__main__":
    main()
