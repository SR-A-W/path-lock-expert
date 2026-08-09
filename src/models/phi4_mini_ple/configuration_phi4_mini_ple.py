# coding=utf-8
# Copyright 2025 the PLE team. All rights reserved.
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
"""Phi-4-mini Path-Lock Expert (PLE) model configuration.

基于 Phi3Config（Phi-4-mini 的 model_type 为 "phi3"），
添加序列级硬路由参数。
"""

from transformers.utils import logging
from transformers.models.phi3.configuration_phi3 import Phi3Config


logger = logging.get_logger(__name__)


class Phi4MiniPleConfig(Phi3Config):
    r"""
    Path-Lock Expert 配置类，继承自 Phi3Config。

    在 Phi3 基础上增加路由相关配置，用于序列级硬路由：
    根据 input_ids 中出现的控制 token 决定整条序列走哪个 expert。

    Phi3/Phi-4-mini 架构特点：
    - MLP 使用融合的 gate_up_proj（gate+up 合并为一个 Linear）+ down_proj
    - Attention 使用融合的 qkv_proj（Q/K/V 合并为一个 Linear）
    - 支持 partial_rotary_factor（仅旋转部分 head_dim）
    - 无 attention bias
    - 有 resid_pdrop（residual dropout）

    路由决策优先级（从高到低）：
    1. 若同时设置了 think_token_id 和 no_think_token_id，以最后出现的控制 token 为准
    2. 若只设置了其中一个，检测该 token 是否出现
    3. 若都未设置或均未出现，使用 default_routing_index

    Args:
        think_token_id (`int`, *optional*):
            think 模式的控制 token ID。检测到时 → routing_index = 1（expert 1 / think）。
        no_think_token_id (`int`, *optional*):
            no_think 模式的控制 token ID。检测到时 → routing_index = 0（expert 0 / no_think）。
        default_routing_index (`int`, *optional*, defaults to 0):
            当未检测到任何控制 token 时的默认路由。
        **kwargs:
            其余参数透传给 `Phi3Config`。
    """

    model_type = "phi4_mini_ple"

    def __init__(
        self,
        think_token_id: int = None,
        no_think_token_id: int = None,
        default_routing_index: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.think_token_id = think_token_id
        self.no_think_token_id = no_think_token_id
        self.default_routing_index = default_routing_index


__all__ = ["Phi4MiniPleConfig"]
