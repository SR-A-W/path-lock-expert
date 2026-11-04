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

Based on Phi3Config (the model_type of Phi-4-mini is "phi3"); adds the
sequence-level hard-routing parameters.
"""

from transformers.utils import logging
from transformers.models.phi3.configuration_phi3 import Phi3Config


logger = logging.get_logger(__name__)


class Phi4MiniPleConfig(Phi3Config):
    r"""
    Path-Lock Expert config class, extending Phi3Config.

    Adds routing options to Phi3 for sequence-level hard routing:
    the control token found in input_ids decides which expert the whole sequence uses.

    Phi3 / Phi-4-mini architecture notes:
    - The MLP uses a fused gate_up_proj (gate and up merged into one Linear) + down_proj.
    - Attention uses a fused qkv_proj (Q/K/V merged into one Linear).
    - Supports partial_rotary_factor (only part of head_dim is rotated).
    - No attention bias.
    - Has resid_pdrop (residual dropout).

    Routing priority (high to low):
    1. If both think_token_id and no_think_token_id are set, the last occurring control token wins.
    2. If only one is set, check whether that token occurs.
    3. If neither is set or none occurs, use default_routing_index.

    Args:
        think_token_id (`int`, *optional*):
            Control token ID for think mode. When detected -> routing_index = 1 (expert 1 / think).
        no_think_token_id (`int`, *optional*):
            Control token ID for no_think mode. When detected -> routing_index = 0 (expert 0 / no_think).
        default_routing_index (`int`, *optional*, defaults to 0):
            Default route when no control token is detected.
        **kwargs:
            Remaining arguments are passed to `Phi3Config`.
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
