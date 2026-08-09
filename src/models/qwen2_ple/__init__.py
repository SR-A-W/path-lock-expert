# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen2 Path-Lock Expert (PLE) 独立版本模块。

import 路径已从相对路径修正为 pip 安装的 transformers 路径。
"""

from .configuration_qwen2_ple import Qwen2PleConfig
from .modeling_qwen2_ple import (
    Qwen2PleExpertMLP,
    Qwen2PleForCausalLM,
    Qwen2PleMLP,
    Qwen2PleModel,
    Qwen2PleDecoderLayer,
    Qwen2PlePreTrainedModel,
)

__all__ = [
    "Qwen2PleConfig",
    "Qwen2PleExpertMLP",
    "Qwen2PleForCausalLM",
    "Qwen2PleMLP",
    "Qwen2PleModel",
    "Qwen2PleDecoderLayer",
    "Qwen2PlePreTrainedModel",
]
