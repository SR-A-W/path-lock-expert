"""PLE 路由与前向传播行为测试（阶段二）—— 4-bit 量化版

加载 PLE 模型（4-bit 量化），验证路由逻辑、生成输出合理性。
参考模型也使用 4-bit 量化加载，确保公平对比。

测试策略：
- 所有模型使用 NF4 4-bit 量化，适配 GPU 显存
- Expert 0 (no_think, Qwen2.5-7B MLP) 应与 Qwen2.5-7B 原始输出一致
- Expert 1 (think, DeepSeek-R1 MLP) 在未 fine-tune 时输出不可读是预期行为
- 重点验证路由机制正确性：routing_index 在 forward/generate 中是否正确传递

运行方式:
    cd ./
    conda activate het
    python -m pytest src/tests/test_ple_weights.py -v -s

注意：需要 bitsandbytes 和 CUDA GPU。
"""

import gc
import json
import os
import sys

import pytest
import torch
import torch.nn.functional as F

# 路径配置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAIN_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "source", "Qwen2.5-7B-Instruct")
MLP_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "source", "DeepSeek-R1-Distill-Qwen-7B")
PLE_PATH = os.path.join(PROJECT_ROOT, "models", "ple_initialized")


# ==================== 工具函数 ====================

def _load_tensor(model_path: str, key: str) -> torch.Tensor:
    """从 safetensors 按需加载单个张量（不加载完整模型）。"""
    from safetensors import safe_open

    index_file = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_file, "r") as f:
        weight_map = json.load(f)["weight_map"]

    shard_file = weight_map[key]
    shard_path = os.path.join(model_path, shard_file)

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _load_model_4bit(model_path, trust_remote_code=False):
    """以 4-bit NF4 量化加载模型到 GPU。"""
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


# ==================== 前提检查 ====================

@pytest.fixture(scope="module", autouse=True)
def check_models_exist():
    """确保所有必需的模型目录和 CUDA 可用。"""
    for path, name in [
        (MAIN_MODEL_PATH, "Qwen2.5-7B-Instruct"),
        (PLE_PATH, "PLE initialized"),
    ]:
        assert os.path.exists(os.path.join(path, "config.json")), (
            f"模型 {name} 不存在: {path}"
        )
    assert torch.cuda.is_available(), "此测试需要 CUDA GPU"


# ==================== 模型 fixtures ====================

@pytest.fixture(scope="module")
def ple_model():
    """加载 PLE 模型（4-bit 量化）。"""
    print("\n[加载 PLE 模型 4-bit...]")
    model = _load_model_4bit(PLE_PATH, trust_remote_code=True)
    print(f"  PLE 加载完成")
    yield model
    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def tokenizer():
    """加载 tokenizer。"""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MAIN_MODEL_PATH)


# ==================== T2.1: forward 路由测试 ====================

class TestForwardRouting:
    """验证 routing_index 在 forward 中正确传递，两条路径产生不同 logits。"""

    def test_routing_index_0_valid(self, ple_model):
        """routing_index=0 应正常产出 logits。"""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(ple_model.device)
        with torch.no_grad():
            out = ple_model(input_ids=input_ids, routing_index=0)
        assert out.logits is not None
        assert out.logits.shape[-1] == 152064

    def test_routing_index_1_valid(self, ple_model):
        """routing_index=1 应正常产出 logits。"""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(ple_model.device)
        with torch.no_grad():
            out = ple_model(input_ids=input_ids, routing_index=1)
        assert out.logits is not None
        assert out.logits.shape[-1] == 152064

    def test_different_routes_produce_different_logits(self, ple_model):
        """routing_index=0 和 =1 的 logits 应不同（两个 expert 的 MLP 权重不同）。"""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(ple_model.device)
        with torch.no_grad():
            logits_r0 = ple_model(input_ids=input_ids, routing_index=0).logits.cpu()
            logits_r1 = ple_model(input_ids=input_ids, routing_index=1).logits.cpu()
        assert not torch.equal(logits_r0, logits_r1), \
            "routing_index=0 和 =1 的 logits 完全相同——MLP 路由未生效"

    def test_default_routing_uses_config(self, ple_model):
        """不传 routing_index 时，应使用 config.default_routing_index 的值。"""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(ple_model.device)
        default_idx = ple_model.config.default_routing_index

        with torch.no_grad():
            logits_default = ple_model(input_ids=input_ids).logits.cpu()
            logits_explicit = ple_model(
                input_ids=input_ids, routing_index=default_idx
            ).logits.cpu()

        assert torch.equal(logits_default, logits_explicit), (
            f"默认路由（不传 routing_index）应等价于 routing_index={default_idx}"
        )


# ==================== T2.2: generate 路由测试 ====================

class TestGenerateRouting:
    """验证 routing_index 在 generate 中正确传递。

    修复前的 Bug：prepare_inputs_for_generation 忽略了用户传入的 routing_index，
    导致 generate 中所有路由都走 default。修复后应可通过以下测试。
    """

    def test_generate_with_explicit_routing_index(self, ple_model, tokenizer):
        """通过 generate(routing_index=N) 显式指定路由应生效。"""
        prompt = "What is 2 + 3? Answer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(ple_model.device)

        with torch.no_grad():
            gen_r0 = ple_model.generate(
                **inputs, max_new_tokens=30, do_sample=False, routing_index=0
            )
            gen_r1 = ple_model.generate(
                **inputs, max_new_tokens=30, do_sample=False, routing_index=1
            )

        text_r0 = tokenizer.decode(gen_r0[0], skip_special_tokens=True)
        text_r1 = tokenizer.decode(gen_r1[0], skip_special_tokens=True)

        print(f"\n  [generate r=0] {text_r0[:200]}")
        print(f"  [generate r=1] {text_r1[:200]}")

        assert text_r0 != text_r1, \
            "generate 中 routing_index=0 和 =1 输出相同——路由参数未正确传递"

    def test_generate_default_matches_expert0(self, ple_model, tokenizer):
        """不传 routing_index 时，generate 应使用 config.default_routing_index=0。"""
        prompt = "Hello, how are you?"
        inputs = tokenizer(prompt, return_tensors="pt").to(ple_model.device)

        with torch.no_grad():
            gen_default = ple_model.generate(
                **inputs, max_new_tokens=30, do_sample=False,
            )
            gen_r0 = ple_model.generate(
                **inputs, max_new_tokens=30, do_sample=False, routing_index=0
            )

        text_default = tokenizer.decode(gen_default[0], skip_special_tokens=True)
        text_r0 = tokenizer.decode(gen_r0[0], skip_special_tokens=True)

        print(f"\n  [generate default] {text_default[:200]}")
        print(f"  [generate r=0]     {text_r0[:200]}")

        assert text_default == text_r0, \
            "默认生成应与 routing_index=0 一致（config.default_routing_index=0）"

    def test_cached_routing_index_correct(self, ple_model, tokenizer):
        """generate 后 _cached_routing_index 应反映实际使用的路由。"""
        prompt = "Test prompt"
        inputs = tokenizer(prompt, return_tensors="pt").to(ple_model.device)

        with torch.no_grad():
            ple_model.generate(**inputs, max_new_tokens=5, do_sample=False, routing_index=1)
        assert ple_model._cached_routing_index == 1

        with torch.no_grad():
            ple_model.generate(**inputs, max_new_tokens=5, do_sample=False, routing_index=0)
        assert ple_model._cached_routing_index == 0


# ==================== T2.3: Expert 0 输出一致性 ====================

class TestExpert0Consistency:
    """验证 PLE (expert 0) 的 generate 输出与 Qwen2.5-7B-Instruct 一致。

    两者使用相同的 4-bit 量化配置，结果应完全一致。
    PLE expert 0 的权重直接来自 Qwen2.5-7B，共享层也来自 Qwen2.5-7B，
    因此 routing_index=0 的模型行为应等价于原始 Qwen2.5-7B。
    """

    @pytest.fixture(scope="class")
    def qwen_model(self):
        """加载 Qwen2.5-7B-Instruct（4-bit 量化）用于对比。"""
        print("\n[加载 Qwen2.5-7B-Instruct 4-bit...]")
        model = _load_model_4bit(MAIN_MODEL_PATH)
        print(f"  Qwen2.5-7B 加载完成")
        yield model
        del model
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.mark.parametrize("prompt", [
        "What is 2 + 3? Answer:",
        "Hello, how are you?",
        "Explain gravity in one sentence.",
    ])
    def test_generate_matches_qwen(self, ple_model, qwen_model, tokenizer, prompt):
        """PLE expert 0 的生成输出应与 Qwen2.5-7B 完全一致。"""
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            gen_pl = ple_model.generate(
                input_ids=inputs["input_ids"].to(ple_model.device),
                attention_mask=inputs["attention_mask"].to(ple_model.device),
                max_new_tokens=50,
                do_sample=False,
                routing_index=0,
            )
            gen_qwen = qwen_model.generate(
                input_ids=inputs["input_ids"].to(qwen_model.device),
                attention_mask=inputs["attention_mask"].to(qwen_model.device),
                max_new_tokens=50,
                do_sample=False,
            )

        text_pl = tokenizer.decode(gen_pl[0], skip_special_tokens=True)
        text_qwen = tokenizer.decode(gen_qwen[0], skip_special_tokens=True)

        print(f"\n  Prompt: {prompt}")
        print(f"  [PLE r=0] {text_pl[:200]}")
        print(f"  [Qwen2.5-7B] {text_qwen[:200]}")

        assert text_pl == text_qwen, (
            f"PLE expert 0 的输出应与 Qwen2.5-7B 一致\n"
            f"  PLE: {text_pl[:100]}\n"
            f"  Qwen:   {text_qwen[:100]}"
        )

    def test_forward_logits_match(self, ple_model, qwen_model, tokenizer):
        """PLE expert 0 的 forward logits 应与 Qwen2.5-7B 完全一致。"""
        prompt = "The answer to life is"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            logits_pl = ple_model(
                input_ids=inputs["input_ids"].to(ple_model.device),
                attention_mask=inputs["attention_mask"].to(ple_model.device),
                routing_index=0,
            ).logits.cpu().float()

            logits_qwen = qwen_model(
                input_ids=inputs["input_ids"].to(qwen_model.device),
                attention_mask=inputs["attention_mask"].to(qwen_model.device),
            ).logits.cpu().float()

        # top-10 预测应一致
        top_pl = logits_pl[0, -1].topk(10).indices.tolist()
        top_qwen = logits_qwen[0, -1].topk(10).indices.tolist()
        print(f"\n  PLE r=0 top10: {top_pl}")
        print(f"  Qwen2.5-7B top10: {top_qwen}")

        assert top_pl == top_qwen, (
            f"PLE expert 0 的 top-10 预测应与 Qwen2.5-7B 一致\n"
            f"  PLE: {top_pl}\n"
            f"  Qwen:   {top_qwen}"
        )


# ==================== T2.4: Embedding 层一致性 ====================

class TestEmbeddingConsistency:
    """验证 PLE 的 Embedding 与 Qwen2.5-7B 一致（共享层，从 safetensors 对比）。"""

    def test_embedding_output(self, ple_model):
        """相同 input_ids → 相同 embedding 输出。"""
        input_ids = torch.tensor([[1, 100, 500, 1000, 2000]])

        # PLE embedding
        pl_embed = ple_model.model.embed_tokens
        pl_embed_device = next(pl_embed.parameters()).device
        ids_pl = input_ids.to(pl_embed_device)
        with torch.no_grad():
            pl_out = pl_embed(ids_pl).float().cpu()

        # 参考：从 safetensors 加载 Qwen embedding 权重并手动计算
        ref_embed_weight = _load_tensor(MAIN_MODEL_PATH, "model.embed_tokens.weight")
        ref_out = F.embedding(input_ids, ref_embed_weight).float()

        assert torch.allclose(pl_out, ref_out, atol=1e-4), (
            f"Embedding 输出不一致, max diff = {(pl_out - ref_out).abs().max().item()}"
        )


# ==================== T2.5: Expert 1 行为测试 ====================

class TestExpert1Behavior:
    """验证 expert 1 的行为特征。

    注意：Expert 1 (DeepSeek-R1 MLP) 在未 fine-tune 时与 Qwen2.5 的共享层
    （attention, norm, embedding）不兼容，输出不可读是预期行为。
    此测试仅验证：
    1. Expert 1 不会导致异常/crash
    2. Expert 1 和 Expert 0 产生不同的输出
    """

    def test_expert1_no_crash(self, ple_model, tokenizer):
        """Expert 1 路由应能正常 generate，不抛出异常。"""
        prompt = "What is AI?"
        inputs = tokenizer(prompt, return_tensors="pt").to(ple_model.device)

        with torch.no_grad():
            gen = ple_model.generate(
                **inputs, max_new_tokens=20, do_sample=False, routing_index=1
            )
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        print(f"\n  [Expert 1] {text[:200]}")
        # 不检查内容质量，仅确保没有 crash
        assert len(text) > 0

    def test_expert1_differs_from_expert0(self, ple_model, tokenizer):
        """Expert 0 和 Expert 1 的生成输出应不同。"""
        prompt = "2 + 2 ="
        inputs = tokenizer(prompt, return_tensors="pt").to(ple_model.device)

        with torch.no_grad():
            gen_r0 = ple_model.generate(
                **inputs, max_new_tokens=20, do_sample=False, routing_index=0
            )
            gen_r1 = ple_model.generate(
                **inputs, max_new_tokens=20, do_sample=False, routing_index=1
            )

        text_r0 = tokenizer.decode(gen_r0[0], skip_special_tokens=True)
        text_r1 = tokenizer.decode(gen_r1[0], skip_special_tokens=True)

        print(f"\n  [Expert 0] {text_r0[:200]}")
        print(f"  [Expert 1] {text_r1[:200]}")

        assert text_r0 != text_r1, "两个 Expert 的输出完全相同——路由未生效"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
