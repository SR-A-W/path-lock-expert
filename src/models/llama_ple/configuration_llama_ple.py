# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""LLaMA Path-Lock Expert (PLE) model configuration."""

from transformers.utils import logging
from transformers.models.llama.configuration_llama import LlamaConfig


logger = logging.get_logger(__name__)


class LlamaPleConfig(LlamaConfig):
    r"""
    Path-Lock Expert 配置类，继承自 LlamaConfig。

    在 LLaMA 基础上增加路由相关配置，用于序列级硬路由：
    根据 input_ids 中出现的控制 token 决定整条序列走哪个 expert。

    与 Qwen3PLEConfig 的主要区别：
    - 继承 LlamaConfig（而非 Qwen3Config）
    - LLaMA attention 无 bias（attention_bias=False），无 q_norm/k_norm
    - LLaMA 无 sliding_window / layer_types
    - LLaMA 3.1 vocab_size=128256（用满），添加控制 token 后需要 resize embedding

    路由决策优先级（从高到低）：
    1. 若同时设置了 think_token_id 和 no_think_token_id，以**最后出现的**控制 token 为准
    2. 若只设置了其中一个，检测该 token 是否出现
    3. 若都未设置或均未出现，使用 default_routing_index

    Args:
        think_token_id (`int`, *optional*):
            think 模式的控制 token ID。检测到时 → routing_index = 1（expert 1 / think）。
        no_think_token_id (`int`, *optional*):
            no_think 模式的控制 token ID。检测到时 → routing_index = 0（expert 0 / no_think）。
        default_routing_index (`int`, *optional*, defaults to 0):
            当未检测到任何控制 token、或未配置任何 token ID 时的默认路由。
            默认为 0（no_think / expert 0），因为初始化阶段共享层来自源模型，
            与 expert 0 的 MLP 完全兼容。fine-tune 完成后可根据需求修改。
        **kwargs:
            其余参数透传给 `LlamaConfig`。
    """

    model_type = "llama_ple"

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


__all__ = ["LlamaPleConfig"]
