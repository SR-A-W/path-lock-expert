#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLaMA PLE 基础功能测试

测试内容：
1. Config 类正确继承 LlamaConfig，PLE 参数可用
2. 路由逻辑 (_determine_routing_index) 的各种场景
3. 模型实例化和 forward pass
4. generate 阶段的 routing_index 缓存机制

运行方式：
    cd .
    conda activate llms
    python -m pytest src/tests/test_llama_ple_basic.py -v -s
"""

import sys
import os
import pytest
import torch

# 将项目根目录加入路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from src.models.llama_ple.configuration_llama_ple import LlamaPleConfig
from src.models.llama_ple.modeling_llama_ple import (
    LlamaPleExpertMLP,
    LlamaPleForCausalLM,
    LlamaPleMLP,
    LlamaPleModel,
    LlamaPleDecoderLayer,
    _determine_routing_index,
)


# ==================== Fixtures ====================

@pytest.fixture
def small_config():
    """创建一个轻量级配置用于测试（避免加载完整模型）"""
    return LlamaPleConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        think_token_id=998,
        no_think_token_id=999,
        default_routing_index=0,
    )


@pytest.fixture
def small_model(small_config):
    """创建轻量级模型实例"""
    model = LlamaPleForCausalLM(small_config)
    model.eval()
    return model


# ==================== Config Tests ====================

class TestLlamaPleConfig:
    def test_model_type(self, small_config):
        assert small_config.model_type == "llama_ple"

    def test_routing_params(self, small_config):
        assert small_config.think_token_id == 998
        assert small_config.no_think_token_id == 999
        assert small_config.default_routing_index == 0

    def test_llama_params_inherited(self, small_config):
        """确认 LlamaConfig 的参数正确继承"""
        assert small_config.hidden_size == 64
        assert small_config.intermediate_size == 128
        assert small_config.num_hidden_layers == 2
        assert small_config.num_attention_heads == 4
        assert small_config.num_key_value_heads == 2
        assert small_config.hidden_act == "silu"
        assert small_config.attention_bias == False
        assert small_config.mlp_bias == False
        assert small_config.tie_word_embeddings == False

    def test_default_routing_none_tokens(self):
        """未设置控制 token 时的默认行为"""
        config = LlamaPleConfig(
            vocab_size=100, hidden_size=64, intermediate_size=128,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        )
        assert config.think_token_id is None
        assert config.no_think_token_id is None
        assert config.default_routing_index == 0


# ==================== Routing Logic Tests ====================

class TestRoutingLogic:
    def test_no_control_tokens_returns_default(self, small_config):
        """无控制 token → 返回 default_routing_index"""
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        assert _determine_routing_index(ids, small_config) == 0

    def test_think_token_routes_to_1(self, small_config):
        """/think token → routing_index = 1"""
        ids = torch.tensor([[1, 2, 998, 4, 5]])
        assert _determine_routing_index(ids, small_config) == 1

    def test_no_think_token_routes_to_0(self, small_config):
        """/no_think token → routing_index = 0"""
        ids = torch.tensor([[1, 2, 999, 4, 5]])
        assert _determine_routing_index(ids, small_config) == 0

    def test_last_control_token_wins_no_think_last(self, small_config):
        """/think 在前, /no_think 在后 → 0（最后出现的优先）"""
        ids = torch.tensor([[998, 2, 3, 999, 5]])
        assert _determine_routing_index(ids, small_config) == 0

    def test_last_control_token_wins_think_last(self, small_config):
        """/no_think 在前, /think 在后 → 1（最后出现的优先）"""
        ids = torch.tensor([[999, 2, 3, 998, 5]])
        assert _determine_routing_index(ids, small_config) == 1

    def test_unconfigured_tokens_use_default(self):
        """未配置任何 token ID → 使用 default_routing_index"""
        config = LlamaPleConfig(
            vocab_size=100, hidden_size=64, intermediate_size=128,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            default_routing_index=1,
        )
        ids = torch.tensor([[1, 2, 3]])
        assert _determine_routing_index(ids, config) == 1

    def test_batch_routing(self, small_config):
        """batch 内多序列 → 以最后出现的控制 token 为准"""
        ids = torch.tensor([
            [1, 998, 3, 4, 5],
            [1, 2, 3, 999, 5],
        ])
        # /no_think (999) 在全局位置 [1,3] 比 /think (998) 在 [0,1] 更靠后
        result = _determine_routing_index(ids, small_config)
        assert result == 0


# ==================== Model Structure Tests ====================

class TestModelStructure:
    def test_expert_mlp_count(self, small_model):
        """每个 decoder layer 应有 2 个 expert MLP"""
        for layer in small_model.model.layers:
            assert hasattr(layer.mlp, 'experts')
            assert len(layer.mlp.experts) == 2
            assert isinstance(layer.mlp.experts[0], LlamaPleExpertMLP)
            assert isinstance(layer.mlp.experts[1], LlamaPleExpertMLP)

    def test_layer_count(self, small_model, small_config):
        assert len(small_model.model.layers) == small_config.num_hidden_layers

    def test_shared_attention(self, small_model):
        """attention 是共享层，不是 expert 分离的"""
        for layer in small_model.model.layers:
            assert hasattr(layer, 'self_attn')
            assert not hasattr(layer.self_attn, 'experts')


# ==================== Forward Pass Tests ====================

class TestForwardPass:
    def test_forward_expert_0(self, small_model):
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            output = small_model(input_ids=ids, routing_index=0)
        assert output.logits.shape == (1, 5, 1000)

    def test_forward_expert_1(self, small_model):
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            output = small_model(input_ids=ids, routing_index=1)
        assert output.logits.shape == (1, 5, 1000)

    def test_forward_auto_routing(self, small_model):
        """不传 routing_index → 自动从 input_ids 检测"""
        ids_think = torch.tensor([[1, 998, 3, 4, 5]])
        with torch.no_grad():
            output = small_model(input_ids=ids_think)
        assert output.logits.shape == (1, 5, 1000)


# ==================== Generate Tests ====================

class TestGenerate:
    def test_generate_with_explicit_routing(self, small_model):
        """generate(routing_index=0) 应缓存 routing_index"""
        ids = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            gen = small_model.generate(ids, max_new_tokens=3, routing_index=0, do_sample=False)
        assert gen.shape[1] > ids.shape[1]  # 确实生成了新 token
        assert small_model._cached_routing_index == 0

    def test_generate_with_routing_1(self, small_model):
        ids = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            gen = small_model.generate(ids, max_new_tokens=3, routing_index=1, do_sample=False)
        assert small_model._cached_routing_index == 1

    def test_generate_auto_detect_think(self, small_model):
        """input 中有 /think token → 自动检测并缓存 routing_index=1"""
        ids = torch.tensor([[1, 998, 3]])
        with torch.no_grad():
            gen = small_model.generate(ids, max_new_tokens=3, do_sample=False)
        assert small_model._cached_routing_index == 1

    def test_generate_auto_detect_no_think(self, small_model):
        ids = torch.tensor([[1, 999, 3]])
        with torch.no_grad():
            gen = small_model.generate(ids, max_new_tokens=3, do_sample=False)
        assert small_model._cached_routing_index == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
