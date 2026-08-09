#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLaMA PLE 权重一致性测试（需要 GPU）

测试内容：
1. 静态权重验证（分片数量、expert 对称性、参数名完整性）
2. 4-bit 量化加载 + 路由测试
3. PLE (expert 0 / default routing) 与原始 LLaMA 3.1-8B-Instruct 的输出对比

运行方式：
    cd .
    conda activate llms
    python -m pytest src/tests/test_llama_ple_weights.py -v -s

注意：需要 GPU 和以下模型文件：
    - models/source/Llama-3.1-8B-Instruct
    - models/llama_ple_initialized
"""

import json
import os
import sys

import pytest
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

# 模型路径
SOURCE_MODEL_PATH = os.path.join(project_root, "models", "source", "Llama-3.1-8B-Instruct")
PLE_MODEL_PATH = os.path.join(project_root, "models", "llama_ple_initialized")

# 跳过条件
skip_no_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="需要 GPU"
)
skip_no_source = pytest.mark.skipif(
    not os.path.exists(SOURCE_MODEL_PATH), reason=f"源模型不存在: {SOURCE_MODEL_PATH}"
)
skip_no_ple = pytest.mark.skipif(
    not os.path.exists(PLE_MODEL_PATH), reason=f"PLE 模型不存在: {PLE_MODEL_PATH}"
)


# ==================== 静态权重验证（无需 GPU） ====================

class TestStaticWeights:
    """静态权重验证，不加载模型，只检查 safetensors 索引"""

    @skip_no_ple
    def test_weight_count(self):
        """验证总权重数 = 32层 × 12/层 + 3全局 = 387"""
        index_path = os.path.join(PLE_MODEL_PATH, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_keys = list(index["weight_map"].keys())
        assert len(weight_keys) == 387, f"Expected 387, got {len(weight_keys)}"

    @skip_no_ple
    def test_expert_symmetry(self):
        """Expert 0 和 Expert 1 的权重名应一一对应"""
        index_path = os.path.join(PLE_MODEL_PATH, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_keys = list(index["weight_map"].keys())

        e0_keys = sorted([k for k in weight_keys if ".experts.0." in k])
        e1_keys = sorted([k for k in weight_keys if ".experts.1." in k])

        assert len(e0_keys) == 96, f"Expert 0: expected 96, got {len(e0_keys)}"
        assert len(e1_keys) == 96, f"Expert 1: expected 96, got {len(e1_keys)}"

        # 名字应该一一对应（只是 .0. vs .1.）
        for k0, k1 in zip(e0_keys, e1_keys):
            assert k0.replace(".experts.0.", ".experts.1.") == k1

    @skip_no_ple
    def test_no_gate_weights(self):
        """PLE 不应有 gate/router 权重"""
        index_path = os.path.join(PLE_MODEL_PATH, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index = json.load(f)
        gate_keys = [k for k in index["weight_map"] if "gate.weight" in k and "gate_proj" not in k]
        assert len(gate_keys) == 0, f"发现异常 gate 权重: {gate_keys}"

    @skip_no_ple
    def test_config_fields(self):
        """验证 config.json 包含 PLE 必要字段"""
        config_path = os.path.join(PLE_MODEL_PATH, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        assert config["model_type"] == "llama_ple"
        assert config["think_token_id"] == 128256
        assert config["no_think_token_id"] == 128257
        assert config["default_routing_index"] == 0
        assert config["vocab_size"] == 128258

    @skip_no_ple
    def test_embedding_shapes(self):
        """验证 embedding 和 lm_head 已 resize 到 128258"""
        from safetensors import safe_open

        index_path = os.path.join(PLE_MODEL_PATH, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index = json.load(f)

        for key in ["model.embed_tokens.weight", "lm_head.weight"]:
            shard = index["weight_map"][key]
            shard_path = os.path.join(PLE_MODEL_PATH, shard)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                t = f.get_tensor(key)
                assert t.shape[0] == 128258, f"{key}: expected [128258, ...], got {list(t.shape)}"
                assert t.shape[1] == 4096


# ==================== GPU 加载与路由测试 ====================

@skip_no_gpu
@skip_no_ple
class TestGPURouting:
    """4-bit 量化加载 + 路由功能测试"""

    @pytest.fixture(scope="class")
    def ple_model(self):
        """加载 4-bit 量化的 PLE 模型（class 级别缓存）"""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print(f"\n加载 PLE 模型 (4-bit): {PLE_MODEL_PATH}")
        model = AutoModelForCausalLM.from_pretrained(
            PLE_MODEL_PATH,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(PLE_MODEL_PATH, trust_remote_code=True)
        model.eval()
        return model, tokenizer

    def test_model_loads(self, ple_model):
        """模型能正常加载"""
        model, tokenizer = ple_model
        assert model is not None
        assert len(tokenizer) == 128258

    def test_generate_expert_0(self, ple_model):
        """Expert 0 (no_think) 能正常生成"""
        model, tokenizer = ple_model
        prompt = "/no_think What is 2+2?"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20, do_sample=False, routing_index=0)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"  Expert 0 output: {text[:100]}")
        assert len(output[0]) > len(inputs["input_ids"][0])

    def test_generate_expert_1(self, ple_model):
        """Expert 1 (think) 能正常生成"""
        model, tokenizer = ple_model
        prompt = "/think What is 2+2?"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20, do_sample=False, routing_index=1)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"  Expert 1 output: {text[:100]}")
        assert len(output[0]) > len(inputs["input_ids"][0])

    def test_routing_cached(self, ple_model):
        """generate 后 routing_index 被正确缓存"""
        model, tokenizer = ple_model
        inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=5, do_sample=False, routing_index=0)
        assert model._cached_routing_index == 0

        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=5, do_sample=False, routing_index=1)
        assert model._cached_routing_index == 1

    def test_auto_routing_from_control_token(self, ple_model):
        """自动从 input 中检测控制 token 并路由"""
        model, tokenizer = ple_model

        # /think → expert 1
        inputs_think = tokenizer("/think Hello", return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**inputs_think, max_new_tokens=5, do_sample=False)
        assert model._cached_routing_index == 1

        # /no_think → expert 0
        inputs_no_think = tokenizer("/no_think Hello", return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**inputs_no_think, max_new_tokens=5, do_sample=False)
        assert model._cached_routing_index == 0


# ==================== 与原始模型的输出对比 ====================

@skip_no_gpu
@skip_no_source
@skip_no_ple
class TestOutputConsistency:
    """PLE (expert 0) 应与原始 LLaMA 3.1 输出一致"""

    @pytest.fixture(scope="class")
    def both_models(self):
        """同时加载原始模型和 PLE（bf16，共享 GPU）"""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        print(f"\n加载原始模型 (4-bit): {SOURCE_MODEL_PATH}")
        source_model = AutoModelForCausalLM.from_pretrained(
            SOURCE_MODEL_PATH,
            quantization_config=quantization_config,
            device_map="auto",
        )
        source_tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL_PATH)
        source_model.eval()

        print(f"加载 PLE 模型 (4-bit): {PLE_MODEL_PATH}")
        ple_model = AutoModelForCausalLM.from_pretrained(
            PLE_MODEL_PATH,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",
        )
        ple_tokenizer = AutoTokenizer.from_pretrained(PLE_MODEL_PATH, trust_remote_code=True)
        ple_model.eval()

        return source_model, source_tokenizer, ple_model, ple_tokenizer

    def test_greedy_output_matches(self, both_models):
        """Expert 0 的 greedy decode 输出应与原始模型逐 token 一致"""
        source_model, source_tok, ple_model, ple_tok = both_models

        # 使用不包含控制 token 的 prompt（走 default routing = expert 0）
        prompt = "The capital of France is"
        source_inputs = source_tok(prompt, return_tensors="pt").to(source_model.device)
        ple_inputs = ple_tok(prompt, return_tensors="pt").to(ple_model.device)

        max_new = 30
        with torch.no_grad():
            source_out = source_model.generate(
                **source_inputs, max_new_tokens=max_new, do_sample=False
            )
            ple_out = ple_model.generate(
                **ple_inputs, max_new_tokens=max_new, do_sample=False, routing_index=0
            )

        source_text = source_tok.decode(source_out[0], skip_special_tokens=True)
        ple_text = ple_tok.decode(ple_out[0], skip_special_tokens=True)

        print(f"\n  Source:  {source_text[:120]}")
        print(f"  PLE:  {ple_text[:120]}")

        # 逐 token 对比
        source_ids = source_out[0].tolist()
        ple_ids = ple_out[0].tolist()

        # 比较生成的新 token（跳过 prompt 部分）
        prompt_len = source_inputs["input_ids"].shape[1]
        source_gen = source_ids[prompt_len:]
        ple_gen = ple_ids[prompt_len:]

        match_count = sum(1 for a, b in zip(source_gen, ple_gen) if a == b)
        total = min(len(source_gen), len(ple_gen))
        match_rate = match_count / total if total > 0 else 0

        print(f"  Token match: {match_count}/{total} ({match_rate:.1%})")
        assert match_rate == 1.0, (
            f"PLE expert 0 输出与原始模型不一致 ({match_rate:.1%} match).\n"
            f"Source tokens: {source_gen[:10]}\n"
            f"PLE tokens: {ple_gen[:10]}"
        )

    def test_logits_close(self, both_models):
        """单步 forward 的 logits 应非常接近"""
        source_model, source_tok, ple_model, ple_tok = both_models

        prompt = "Hello world"
        source_inputs = source_tok(prompt, return_tensors="pt").to(source_model.device)
        ple_inputs = ple_tok(prompt, return_tensors="pt").to(ple_model.device)

        with torch.no_grad():
            source_logits = source_model(**source_inputs).logits
            ple_logits = ple_model(**ple_inputs, routing_index=0).logits

        # 只比较原始 vocab 范围内的 logits（前 128256 个）
        source_logits_trimmed = source_logits[:, :, :128256]
        ple_logits_trimmed = ple_logits[:, :, :128256]

        # 检查 argmax 是否一致
        source_argmax = source_logits_trimmed.argmax(dim=-1)
        ple_argmax = ple_logits_trimmed.argmax(dim=-1)
        argmax_match = (source_argmax == ple_argmax).float().mean().item()
        print(f"\n  Argmax match rate: {argmax_match:.1%}")
        assert argmax_match == 1.0, f"Argmax 不完全一致: {argmax_match:.1%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
