"""PLE 静态权重一致性检查（阶段一）

直接从 safetensors 文件逐张量比较，不加载完整模型到内存。
验证权重复制脚本 (copy_weights_to_ple.py) 的产出正确性。

运行方式:
    cd ./
    conda activate het
    python -m pytest src/tests/test_ple_weights_lite.py -v -s

预期耗时：~1-2 分钟（逐张量流式读取比较，内存占用极低）
"""

import json
import os
import sys

import pytest
import torch

# 路径配置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAIN_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "source", "Qwen2.5-7B-Instruct")
MLP_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "source", "DeepSeek-R1-Distill-Qwen-7B")
PLE_PATH = os.path.join(PROJECT_ROOT, "models", "ple_initialized")

# 模型参数
NUM_LAYERS = 28
MLP_PROJS = ["gate_proj", "up_proj", "down_proj"]


# ==================== 工具函数 ====================

def _load_tensor(model_path: str, key: str) -> torch.Tensor:
    """从 safetensors 中按需加载单个张量（内存友好）。

    先读取 index.json 确定分片文件，再用 safe_open 打开该分片。

    Args:
        model_path: 模型目录路径
        key: 权重键名

    Returns:
        CPU 上的权重张量
    """
    from safetensors import safe_open

    index_file = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_file, "r") as f:
        weight_map = json.load(f)["weight_map"]

    shard_file = weight_map[key]
    shard_path = os.path.join(model_path, shard_file)

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _get_all_keys(model_path: str) -> list:
    """获取模型的所有权重键名。"""
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_file, "r") as f:
        return sorted(json.load(f)["weight_map"].keys())


# ==================== 前提检查 ====================

@pytest.fixture(scope="module", autouse=True)
def check_models_exist():
    """确保所有必需的模型目录存在。"""
    for path, name in [
        (MAIN_MODEL_PATH, "Qwen2.5-7B-Instruct"),
        (MLP_MODEL_PATH, "DeepSeek-R1-Distill-Qwen-7B"),
        (PLE_PATH, "PLE initialized"),
    ]:
        index = os.path.join(path, "model.safetensors.index.json")
        assert os.path.exists(index), (
            f"模型 {name} 不存在或未完成初始化: {path}\n"
            f"请先运行: python -m src.utils.copy_weights_to_ple ..."
        )


# ==================== P-T1.1: Expert 0 MLP 权重 == Qwen2.5-7B MLP ====================

class TestExpert0MLP:
    """验证 Expert 0 (no_think) 的 MLP 权重与 Qwen2.5-7B-Instruct 完全一致。"""

    @pytest.mark.parametrize("layer_idx", range(NUM_LAYERS))
    @pytest.mark.parametrize("proj", MLP_PROJS)
    def test_expert0_mlp_weight(self, layer_idx, proj):
        """逐层、逐投影对比 Expert 0 MLP 权重与 Qwen2.5-7B MLP 权重。"""
        src_key = f"model.layers.{layer_idx}.mlp.{proj}.weight"
        tgt_key = f"model.layers.{layer_idx}.mlp.experts.0.{proj}.weight"

        src_tensor = _load_tensor(MAIN_MODEL_PATH, src_key)
        tgt_tensor = _load_tensor(PLE_PATH, tgt_key)

        assert src_tensor.shape == tgt_tensor.shape, (
            f"形状不匹配: {src_key} {src_tensor.shape} vs {tgt_key} {tgt_tensor.shape}"
        )
        assert torch.equal(src_tensor, tgt_tensor), (
            f"权重不一致: layer {layer_idx} {proj}\n"
            f"  max diff = {(src_tensor - tgt_tensor).abs().max().item()}"
        )


# ==================== P-T1.2: Expert 1 MLP 权重 == DeepSeek-R1 MLP ====================

class TestExpert1MLP:
    """验证 Expert 1 (think) 的 MLP 权重与 DeepSeek-R1-Distill-Qwen-7B 完全一致。"""

    @pytest.mark.parametrize("layer_idx", range(NUM_LAYERS))
    @pytest.mark.parametrize("proj", MLP_PROJS)
    def test_expert1_mlp_weight(self, layer_idx, proj):
        """逐层、逐投影对比 Expert 1 MLP 权重与 DeepSeek-R1 MLP 权重。"""
        src_key = f"model.layers.{layer_idx}.mlp.{proj}.weight"
        tgt_key = f"model.layers.{layer_idx}.mlp.experts.1.{proj}.weight"

        src_tensor = _load_tensor(MLP_MODEL_PATH, src_key)
        tgt_tensor = _load_tensor(PLE_PATH, tgt_key)

        assert src_tensor.shape == tgt_tensor.shape, (
            f"形状不匹配: {src_key} {src_tensor.shape} vs {tgt_key} {tgt_tensor.shape}"
        )
        assert torch.equal(src_tensor, tgt_tensor), (
            f"权重不一致: layer {layer_idx} {proj}\n"
            f"  max diff = {(src_tensor - tgt_tensor).abs().max().item()}"
        )


# ==================== P-T1.3: 非 MLP 权重 == Qwen2.5-7B ====================

class TestSharedWeights:
    """验证非 MLP 权重（embedding, attention, layernorm, lm_head）与 Qwen2.5-7B 完全一致。"""

    @pytest.fixture(scope="class")
    def shared_keys(self):
        """获取所有非 MLP 的共享权重键名。

        共享权重 = PLE 中不含 '.experts.' 的权重，
        这些权重在源模型和 PLE 中应该有相同的键名。
        """
        all_keys = _get_all_keys(PLE_PATH)
        return [k for k in all_keys if ".experts." not in k]

    def test_shared_weight_count(self, shared_keys):
        """共享权重数量应等于 Qwen2.5-7B 总权重数 - MLP 权重数。

        Qwen2.5-7B: 339 个权重
        其中每层 3 个 MLP 权重 × 28 层 = 84 个 MLP 权重
        共享权重 = 339 - 84 = 255
        """
        expected = 339 - NUM_LAYERS * len(MLP_PROJS)  # 339 - 84 = 255
        assert len(shared_keys) == expected, (
            f"共享权重数量: {len(shared_keys)}, 预期: {expected}"
        )

    def test_embed_tokens(self):
        """embed_tokens 权重一致性。"""
        key = "model.embed_tokens.weight"
        src = _load_tensor(MAIN_MODEL_PATH, key)
        tgt = _load_tensor(PLE_PATH, key)
        assert torch.equal(src, tgt), f"{key} 不一致"

    def test_final_norm(self):
        """最终 LayerNorm 权重一致性。"""
        key = "model.norm.weight"
        src = _load_tensor(MAIN_MODEL_PATH, key)
        tgt = _load_tensor(PLE_PATH, key)
        assert torch.equal(src, tgt), f"{key} 不一致"

    def test_lm_head(self):
        """lm_head 权重一致性。"""
        key = "lm_head.weight"
        src = _load_tensor(MAIN_MODEL_PATH, key)
        tgt = _load_tensor(PLE_PATH, key)
        assert torch.equal(src, tgt), f"{key} 不一致"

    @pytest.mark.parametrize("layer_idx", range(NUM_LAYERS))
    def test_attention_weights(self, layer_idx):
        """每层 Attention 权重一致性（q, k, v, o 的 weight + bias）。"""
        attn_keys = [
            f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            f"model.layers.{layer_idx}.self_attn.q_proj.bias",
            f"model.layers.{layer_idx}.self_attn.k_proj.weight",
            f"model.layers.{layer_idx}.self_attn.k_proj.bias",
            f"model.layers.{layer_idx}.self_attn.v_proj.weight",
            f"model.layers.{layer_idx}.self_attn.v_proj.bias",
            f"model.layers.{layer_idx}.self_attn.o_proj.weight",
        ]
        for key in attn_keys:
            src = _load_tensor(MAIN_MODEL_PATH, key)
            tgt = _load_tensor(PLE_PATH, key)
            assert torch.equal(src, tgt), (
                f"{key} 不一致, max diff = {(src - tgt).abs().max().item()}"
            )

    @pytest.mark.parametrize("layer_idx", range(NUM_LAYERS))
    def test_layernorm_weights(self, layer_idx):
        """每层 LayerNorm 权重一致性。"""
        norm_keys = [
            f"model.layers.{layer_idx}.input_layernorm.weight",
            f"model.layers.{layer_idx}.post_attention_layernorm.weight",
        ]
        for key in norm_keys:
            src = _load_tensor(MAIN_MODEL_PATH, key)
            tgt = _load_tensor(PLE_PATH, key)
            assert torch.equal(src, tgt), f"{key} 不一致"


# ==================== P-T1.4: 无 Router/gate 权重 ====================

class TestNoGateWeights:
    """确认 PLE 模型中无 gate/router 权重，路由完全由控制 token 决定。"""

    def test_no_gate_keys(self):
        """权重索引中不应包含任何 gate 权重（gate_proj 除外，那是 MLP 的一部分）。"""
        all_keys = _get_all_keys(PLE_PATH)
        gate_keys = [k for k in all_keys if "gate.weight" in k and "gate_proj" not in k]
        assert len(gate_keys) == 0, (
            f"发现异常 gate/router 权重: {gate_keys}\n"
            f"PLE 应无 gate 权重，路由由控制 token 决定"
        )

    def test_no_router_keys(self):
        """权重索引中不应包含 router 相关键名。"""
        all_keys = _get_all_keys(PLE_PATH)
        router_keys = [k for k in all_keys if "router" in k.lower()]
        assert len(router_keys) == 0, f"发现异常 router 权重: {router_keys}"

    def test_total_weight_count(self):
        """总权重数量验证：
        255 共享 + 84 Expert 0 + 84 Expert 1 = 423
        """
        all_keys = _get_all_keys(PLE_PATH)
        assert len(all_keys) == 423, f"总权重数量: {len(all_keys)}, 预期: 423"


# ==================== P-T1.5: 配置文件验证 ====================

class TestConfig:
    """验证输出的 config.json 正确性。"""

    @pytest.fixture(scope="class")
    def config(self):
        with open(os.path.join(PLE_PATH, "config.json"), "r") as f:
            return json.load(f)

    def test_model_type(self, config):
        assert config["model_type"] == "qwen2_ple"

    def test_architectures(self, config):
        assert config["architectures"] == ["Qwen2PleForCausalLM"]

    def test_hidden_size(self, config):
        assert config["hidden_size"] == 3584

    def test_intermediate_size(self, config):
        assert config["intermediate_size"] == 18944

    def test_num_layers(self, config):
        assert config["num_hidden_layers"] == 28

    def test_default_routing_index(self, config):
        assert config["default_routing_index"] == 1

    def test_auto_map(self, config):
        assert "auto_map" in config
        assert "AutoConfig" in config["auto_map"]
        assert "AutoModelForCausalLM" in config["auto_map"]

    def test_no_moe_gate_params(self, config):
        """PLE 配置中不应有 TL-MoE 特有的 gate 参数。"""
        for key in ["num_experts", "num_experts_per_tok", "decoder_sparse_step",
                     "router_aux_loss_coef", "norm_topk_prob"]:
            assert key not in config, f"PLE 配置不应包含: {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
