#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TL-MoE 模型创建辅助工具

提供创建空 TL-MoE 模型配置和实例的辅助函数。
主要用于测试和开发目的。

使用方法:
    from src.utils.create_tl_moe_model import create_tl_moe_config, create_tl_moe_model
    
    # 创建配置（基于 Qwen2.5-7B 规模）
    config = create_tl_moe_config()
    
    # 创建模型实例（随机初始化）
    model = create_tl_moe_model(config)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def create_tl_moe_config(
    hidden_size: int = 3584,
    intermediate_size: int = 18944,
    num_hidden_layers: int = 28,
    num_attention_heads: int = 28,
    num_key_value_heads: int = 4,
    vocab_size: int = 152064,
    **kwargs
):
    """创建 TL-MoE 配置
    
    默认参数与 Qwen2.5-7B-Instruct 一致。
    
    Args:
        hidden_size: 隐藏维度大小
        intermediate_size: MLP 中间维度
        num_hidden_layers: Transformer 层数
        num_attention_heads: 注意力头数
        num_key_value_heads: KV 注意力头数（GQA）
        vocab_size: 词表大小
        **kwargs: 其他可选参数
        
    Returns:
        Qwen2TLMoeConfig 实例
    """
    from src.models.qwen2_tl_moe import Qwen2TLMoeConfig
    
    config = Qwen2TLMoeConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        moe_intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        vocab_size=vocab_size,
        **kwargs
    )
    
    return config


def create_tl_moe_model(config=None, **kwargs):
    """创建 TL-MoE 模型实例（随机初始化）
    
    Args:
        config: 可选的配置对象。如果不提供，使用默认配置。
        **kwargs: 传递给 create_tl_moe_config 的参数
        
    Returns:
        Qwen2TLMoeForCausalLM 实例
    """
    from src.models.qwen2_tl_moe import Qwen2TLMoeForCausalLM
    
    if config is None:
        config = create_tl_moe_config(**kwargs)
    
    model = Qwen2TLMoeForCausalLM(config)
    return model
