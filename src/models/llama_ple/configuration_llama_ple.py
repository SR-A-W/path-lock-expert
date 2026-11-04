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
    Path-Lock Expert config class, extending LlamaConfig.

    Adds routing options to LLaMA for sequence-level hard routing:
    the control token found in input_ids decides which expert the whole sequence uses.

    Differences from Qwen3PLEConfig:
    - Inherits LlamaConfig (not Qwen3Config).
    - LLaMA attention has no bias (attention_bias=False) and no q_norm/k_norm.
    - LLaMA has no sliding_window / layer_types.
    - LLaMA 3.1 uses the full vocab_size=128256, so the embedding must be resized after adding control tokens.

    Routing priority (high to low):
    1. If both think_token_id and no_think_token_id are set, the **last occurring** control token wins.
    2. If only one is set, check whether that token occurs.
    3. If neither is set or none occurs, use default_routing_index.

    Args:
        think_token_id (`int`, *optional*):
            Control token ID for think mode. When detected -> routing_index = 1 (expert 1 / think).
        no_think_token_id (`int`, *optional*):
            Control token ID for no_think mode. When detected -> routing_index = 0 (expert 0 / no_think).
        default_routing_index (`int`, *optional*, defaults to 0):
            Default route when no control token is detected or no token id is configured.
            Defaults to 0 (no_think / expert 0): at initialisation the shared layers come from the
            source model, which is fully compatible with the expert-0 MLP.
        **kwargs:
            Remaining arguments are passed to `LlamaConfig`.
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
