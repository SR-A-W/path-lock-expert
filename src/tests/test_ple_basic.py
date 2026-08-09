"""PLE 模型基本功能测试：配置、创建、前向传播、路由逻辑、generate 路由缓存。

运行方式:
    cd ./
    conda activate het
    python -m pytest src/tests/test_ple_basic.py -v -s
"""

import sys
import os
import torch
import pytest

# 将 transformers submodule 的 src 加入 path，优先于 pip 安装的 transformers
TRANSFORMERS_SRC = os.path.join(
    os.path.dirname(__file__), "..", "transformers", "src"
)
sys.path.insert(0, os.path.abspath(TRANSFORMERS_SRC))

from transformers.models.qwen2_ple.configuration_qwen2_ple import Qwen2PleConfig
from transformers.models.qwen2_ple.modeling_qwen2_ple import (
    Qwen2PleExpertMLP,
    Qwen2PleMLP,
    Qwen2PleDecoderLayer,
    Qwen2PleModel,
    Qwen2PleForCausalLM,
    _determine_routing_index,
)


# ---------- 小型测试用配置（不需要大显存） ----------
THINK_ID = 42
NO_THINK_ID = 43

def _tiny_config(think_token_id=THINK_ID, no_think_token_id=NO_THINK_ID, default_routing_index=1):
    """创建一个极小的 PLE 配置，用于快速 CPU 测试。"""
    return Qwen2PleConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        think_token_id=think_token_id,
        no_think_token_id=no_think_token_id,
        default_routing_index=default_routing_index,
    )


# ===================== 配置测试 =====================
class TestConfig:
    def test_model_type(self):
        """验证 model_type 是 qwen2_ple"""
        config = _tiny_config()
        assert config.model_type == "qwen2_ple"

    def test_think_token_id(self):
        config = _tiny_config(think_token_id=42)
        assert config.think_token_id == 42

    def test_no_think_token_id(self):
        config = _tiny_config(no_think_token_id=43)
        assert config.no_think_token_id == 43

    def test_default_routing_index(self):
        """默认路由是 1（think 模式）"""
        config = _tiny_config()
        assert config.default_routing_index == 1

    def test_default_routing_index_custom(self):
        config = _tiny_config(default_routing_index=0)
        assert config.default_routing_index == 0


# ===================== 路由判断函数测试 =====================
class TestDetermineRouting:
    def test_only_think_token(self):
        """只包含 think token → routing_index = 1"""
        config = _tiny_config()
        ids = torch.tensor([[1, 2, THINK_ID, 3, 4]])
        assert _determine_routing_index(ids, config) == 1

    def test_only_no_think_token(self):
        """只包含 no_think token → routing_index = 0"""
        config = _tiny_config()
        ids = torch.tensor([[1, 2, NO_THINK_ID, 3, 4]])
        assert _determine_routing_index(ids, config) == 0

    def test_no_control_token(self):
        """无控制 token → 使用 default_routing_index (默认 1)"""
        config = _tiny_config()
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        assert _determine_routing_index(ids, config) == 1  # default=1 (think)

    def test_no_control_token_custom_default(self):
        """自定义 default=0 时无控制 token → 返回 0"""
        config = _tiny_config(default_routing_index=0)
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        assert _determine_routing_index(ids, config) == 0

    def test_both_tokens_think_last(self):
        """同时包含两种 token，think 在后 → routing_index = 1"""
        config = _tiny_config()
        ids = torch.tensor([[1, NO_THINK_ID, 2, THINK_ID, 3]])
        assert _determine_routing_index(ids, config) == 1

    def test_both_tokens_no_think_last(self):
        """同时包含两种 token，no_think 在后 → routing_index = 0（防注入）"""
        config = _tiny_config()
        ids = torch.tensor([[1, THINK_ID, 2, NO_THINK_ID, 3]])
        assert _determine_routing_index(ids, config) == 0

    def test_injection_defense(self):
        """模拟注入防御：用户在前面插入了 think token，但系统在后面放了 no_think token"""
        config = _tiny_config()
        # 序列：[bos, user_think_injection, ..., system_no_think_control]
        ids = torch.tensor([[1, THINK_ID, 10, 11, 12, NO_THINK_ID]])
        assert _determine_routing_index(ids, config) == 0  # 最后的 no_think 覆盖注入

    def test_no_token_ids_configured(self):
        """未配置任何 token ID → 直接返回 default"""
        config = _tiny_config(think_token_id=None, no_think_token_id=None, default_routing_index=1)
        ids = torch.tensor([[1, 2, 3]])
        assert _determine_routing_index(ids, config) == 1

    def test_only_think_id_configured_not_found(self):
        """只配置了 think_token_id 但 input_ids 中没有 → default"""
        config = _tiny_config(think_token_id=THINK_ID, no_think_token_id=None, default_routing_index=1)
        ids = torch.tensor([[1, 2, 3]])
        assert _determine_routing_index(ids, config) == 1  # 未找到 → default

    def test_only_think_id_configured_found(self):
        """只配置了 think_token_id 且找到 → 1"""
        config = _tiny_config(think_token_id=THINK_ID, no_think_token_id=None)
        ids = torch.tensor([[1, THINK_ID, 3]])
        assert _determine_routing_index(ids, config) == 1


# ===================== MLP 测试 =====================
class TestMLP:
    def test_expert_mlp_output_shape(self):
        """单个 Expert MLP 的输出形状正确"""
        config = _tiny_config()
        mlp = Qwen2PleExpertMLP(config)
        x = torch.randn(2, 8, 64)
        out = mlp(x)
        assert out.shape == (2, 8, 64)

    def test_ple_mlp_routing(self):
        """PLE MLP 容器中两个 expert 的权重不同，路由到不同 expert 产生不同输出。"""
        config = _tiny_config()
        mlp = Qwen2PleMLP(config)
        x = torch.randn(2, 8, 64)
        out0 = mlp(x, routing_index=0)
        out1 = mlp(x, routing_index=1)
        assert out0.shape == out1.shape == (2, 8, 64)
        assert not torch.allclose(out0, out1), "两个 expert 输出不应完全一致"

    def test_ple_mlp_has_two_experts(self):
        config = _tiny_config()
        mlp = Qwen2PleMLP(config)
        assert len(mlp.experts) == 2


# ===================== Decoder Layer 测试 =====================
class TestDecoderLayer:
    def _make_layer(self):
        config = _tiny_config()
        config._attn_implementation = "eager"
        return config, Qwen2PleDecoderLayer(config, layer_idx=0)

    def test_forward_shape(self):
        config, layer = self._make_layer()
        x = torch.randn(1, 8, 64)
        cos = torch.ones(1, 8, 16)
        sin = torch.zeros(1, 8, 16)
        out = layer(x, position_embeddings=(cos, sin), routing_index=0)
        assert out.shape == (1, 8, 64)

    def test_routing_index_passed(self):
        config, layer = self._make_layer()
        x = torch.randn(1, 8, 64)
        cos = torch.ones(1, 8, 16)
        sin = torch.zeros(1, 8, 16)
        out0 = layer(x, position_embeddings=(cos, sin), routing_index=0)
        out1 = layer(x, position_embeddings=(cos, sin), routing_index=1)
        assert not torch.allclose(out0, out1), "不同 expert 路由应产生不同输出"


# ===================== Model 测试 =====================
class TestModel:
    def test_model_creation(self):
        config = _tiny_config()
        model = Qwen2PleModel(config)
        assert len(model.layers) == 2
        for layer in model.layers:
            assert isinstance(layer.mlp, Qwen2PleMLP)

    def test_forward_with_think(self):
        """包含 think token → routing_index=1"""
        config = _tiny_config()
        model = Qwen2PleModel(config).eval()
        input_ids = torch.tensor([[1, THINK_ID, 3, 4, 5]])
        output = model(input_ids=input_ids)
        assert output.last_hidden_state.shape == (1, 5, 64)

    def test_forward_with_no_think(self):
        """包含 no_think token → routing_index=0"""
        config = _tiny_config()
        model = Qwen2PleModel(config).eval()
        input_ids = torch.tensor([[1, NO_THINK_ID, 3, 4, 5]])
        output = model(input_ids=input_ids)
        assert output.last_hidden_state.shape == (1, 5, 64)

    def test_forward_no_control_token_defaults_to_think(self):
        """无控制 token → 默认走 think (expert 1)，通过显式传 routing_index 验证"""
        config = _tiny_config()
        model = Qwen2PleModel(config).eval()
        # 使用相同的 input_ids，只改变路由方式
        ids = torch.tensor([[1, 2, 3, 4, 5]])  # 不含任何控制 token
        with torch.no_grad():
            # 不传 routing_index → 应自动使用 default_routing_index=1 (think)
            out_default = model(input_ids=ids).last_hidden_state
            # 显式传 routing_index=1 → 手动指定 think
            out_explicit_think = model(input_ids=ids, routing_index=1).last_hidden_state
        # 两者应完全一致：默认路由 = 显式 think
        assert torch.allclose(out_default, out_explicit_think), "default 应与显式 routing_index=1 一致"

    def test_routing_differs(self):
        """think vs no_think 路由输出不同"""
        config = _tiny_config()
        model = Qwen2PleModel(config).eval()
        ids_think = torch.tensor([[1, THINK_ID, 3, 4, 5]])
        ids_no_think = torch.tensor([[1, NO_THINK_ID, 3, 4, 5]])
        with torch.no_grad():
            out_think = model(input_ids=ids_think).last_hidden_state
            out_no_think = model(input_ids=ids_no_think).last_hidden_state
        assert not torch.allclose(out_think, out_no_think), "think vs no_think 输出应不同"

    def test_explicit_routing_index_override(self):
        """手动传入 routing_index 应覆盖自动检测"""
        config = _tiny_config()
        model = Qwen2PleModel(config).eval()
        # 使用相同的 input_ids（含 think token），只靠 routing_index 参数区分
        ids = torch.tensor([[1, THINK_ID, 3, 4, 5]])
        with torch.no_grad():
            # 自动检测 → routing_index=1 (因为有 think token)
            out_auto = model(input_ids=ids).last_hidden_state
            # 手动传 routing_index=0 → 应覆盖自动检测，走 no_think
            out_override = model(input_ids=ids, routing_index=0).last_hidden_state
            # 手动传 routing_index=1 → 显式 think
            out_explicit = model(input_ids=ids, routing_index=1).last_hidden_state
        # 自动 == 显式 think
        assert torch.allclose(out_auto, out_explicit), "自动检测应与显式 routing_index=1 一致"
        # 覆盖 != 自动（不同 expert）
        assert not torch.allclose(out_override, out_auto), "手动 routing_index=0 应与自动 think 不同"


# ===================== ForCausalLM 测试 =====================
class TestForCausalLM:
    def test_creation(self):
        config = _tiny_config()
        model = Qwen2PleForCausalLM(config)
        assert model is not None
        assert model._cached_routing_index is None  # 初始状态

    def test_forward_logits(self):
        config = _tiny_config()
        model = Qwen2PleForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            output = model(input_ids=input_ids)
        assert output.logits.shape == (1, 5, 256)

    def test_forward_with_labels(self):
        config = _tiny_config()
        model = Qwen2PleForCausalLM(config)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        labels = torch.tensor([[2, 3, 4, 5, 0]])
        output = model(input_ids=input_ids, labels=labels)
        assert output.loss is not None
        assert output.loss.ndim == 0

    def test_generate(self):
        """generate 可以正常运行"""
        config = _tiny_config()
        config.pad_token_id = 0
        config.eos_token_id = 255
        model = Qwen2PleForCausalLM(config).eval()
        input_ids = torch.tensor([[1, THINK_ID, 3]])
        with torch.no_grad():
            gen_ids = model.generate(input_ids=input_ids, max_new_tokens=5, do_sample=False)
        assert gen_ids.shape[0] == 1
        assert gen_ids.shape[1] >= 3

    def test_generate_caches_routing(self):
        """generate 过程中 routing_index 应被缓存，不因后续 token 丢失"""
        config = _tiny_config()
        config.pad_token_id = 0
        config.eos_token_id = 255
        model = Qwen2PleForCausalLM(config).eval()

        # 用含 think token 的输入 generate
        input_ids_think = torch.tensor([[1, THINK_ID, 3]])
        with torch.no_grad():
            model.generate(input_ids=input_ids_think, max_new_tokens=3, do_sample=False)
        assert model._cached_routing_index == 1, "生成后应缓存 routing_index=1 (think)"

        # 用含 no_think token 的输入 generate
        input_ids_no_think = torch.tensor([[1, NO_THINK_ID, 3]])
        with torch.no_grad():
            model.generate(input_ids=input_ids_no_think, max_new_tokens=3, do_sample=False)
        assert model._cached_routing_index == 0, "生成后应缓存 routing_index=0 (no_think)"

    def test_generate_different_routes_differ(self):
        """think 和 no_think 路由的 generate 输出应不同"""
        config = _tiny_config()
        config.pad_token_id = 0
        config.eos_token_id = 255
        model = Qwen2PleForCausalLM(config).eval()

        input_ids_think = torch.tensor([[1, THINK_ID, 3]])
        input_ids_no_think = torch.tensor([[1, NO_THINK_ID, 3]])
        with torch.no_grad():
            gen_think = model.generate(input_ids=input_ids_think, max_new_tokens=5, do_sample=False)
            gen_no_think = model.generate(input_ids=input_ids_no_think, max_new_tokens=5, do_sample=False)
        # 两条路由的生成结果不应完全一致（不同 expert 的权重不同）
        assert not torch.equal(gen_think, gen_no_think), "think vs no_think 的 generate 输出应不同"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
