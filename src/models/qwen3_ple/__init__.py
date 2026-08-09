# coding=utf-8
# Copyright 2026 Hybrid-Expert-Thinking Project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Qwen3 PLE (Path-Lock Expert) 独立版本模块。

Qwen3 PLE 实现。import 路径为 pip 安装的 transformers 路径，可独立使用。

路由检测规则：anchor + tokenizer.decode + regex negative-lookbehind + last-occurrence-wins + safe-mode。
详见 modeling_qwen3_ple.py 顶部 docstring 与配套 plan 文档。
"""

from .configuration_qwen3_ple import Qwen3PLEConfig
from .modeling_qwen3_ple import (
    Qwen3PLEDecoderLayer,
    Qwen3PLEExpertMLP,
    Qwen3PLEForCausalLM,
    Qwen3PLEMLP,
    Qwen3PLEModel,
    Qwen3PLEPreTrainedModel,
)

__all__ = [
    "Qwen3PLEConfig",
    "Qwen3PLEDecoderLayer",
    "Qwen3PLEExpertMLP",
    "Qwen3PLEForCausalLM",
    "Qwen3PLEMLP",
    "Qwen3PLEModel",
    "Qwen3PLEPreTrainedModel",
]
