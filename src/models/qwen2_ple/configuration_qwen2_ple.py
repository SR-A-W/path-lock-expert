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
"""Qwen2 Path-Lock Expert (PLE) model configuration."""

from transformers.utils import logging
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


logger = logging.get_logger(__name__)


class Qwen2PleConfig(Qwen2Config):
    r"""
    Path-Lock Expert config class, extending Qwen2Config.

    Adds routing options to Qwen2 for sequence-level hard routing:
    the control token found in input_ids decides which expert the whole sequence uses.

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
            Defaults to 0 (no_think / expert 0): at initialisation the shared layers come from
            Qwen2.5-7B-Instruct, which is fully compatible with the expert-0 MLP.
        **kwargs:
            Remaining arguments are passed to `Qwen2Config`.
    """

    model_type = "qwen2_ple"

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


__all__ = ["Qwen2PleConfig"]
