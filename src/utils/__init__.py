# -*- coding: utf-8 -*-
"""TL-MoE 工具模块

包含：
- create_tl_moe_model: 创建空的 TL-MoE 模型
- copy_weights_to_moe: 复制权重到 TL-MoE 模型（内存优化版本）
- download_source_models: 下载源模型权重
"""

from .create_tl_moe_model import create_tl_moe_model, create_tl_moe_config
from .copy_weights_to_moe import (
    copy_main_model_weights,
    copy_mlp_model_weights,
    verify_gate_weights,
    save_model,
    check_model_compatibility,
    load_state_dict_from_path,
)
from .download_source_models import download_model, verify_model

__all__ = [
    "create_tl_moe_model",
    "create_tl_moe_config",
    "copy_main_model_weights",
    "copy_mlp_model_weights",
    "verify_gate_weights",
    "save_model",
    "check_model_compatibility",
    "load_state_dict_from_path",
    "download_model",
    "verify_model",
]
