# WARNING: This file doesn't work here. It is copied from transformers/src/transformers/models/qwen2_ple/modular_qwen2_ple.py.
# Have to put it in transformers subdirectory and use transformer's converter to generate modeling_qwen2_ple.py. Otherwise, this code will not work.
# This is just an example of how to implement PLE in transformers, and show its structure.


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
"""PyTorch Qwen2 Path-Lock Expert (PLE) model.

Sequence-level routing: the control token (/think or /no_think) fixes expert 0 or expert 1
for the whole sequence. There is no gate network; routing is decided by the control token.

Routing rules:
- expert 0 = no_think mode (initialised from Qwen2.5-7B-Instruct)
- expert 1 = think mode (initialised from DeepSeek-R1-Distill-Qwen-7B)
- The default is expert 0 (no_think), since at initialisation the shared layers come from Qwen2.5 and are compatible with expert 0.
- default_routing_index can be changed after fine-tuning if needed.

Note: all sequences in a batch are assumed to share the same route (a single routing_index).
Mixed routing within a batch (different experts per sequence) would need per-sample routing.
"""

from typing import Optional

import torch
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.generic import check_model_inputs
from ..qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2Model,
    Qwen2MLP,
)
from .configuration_qwen2_ple import Qwen2PleConfig


class Qwen2PleExpertMLP(Qwen2MLP):
    """A single MLP expert (identical in structure to Qwen2MLP), used in the PLE two-expert container."""
    pass


class Qwen2PleMLP(nn.Module):
    """Path-Lock two-expert MLP container.

    Holds two independent MLP experts:
    - experts[0]: no_think expert
    - experts[1]: think expert
    routing_index selects which expert is used; the whole sequence uses the same one.
    """

    def __init__(self, config: Qwen2PleConfig):
        super().__init__()
        self.experts = nn.ModuleList([Qwen2PleExpertMLP(config) for _ in range(2)])

    def forward(self, hidden_states: torch.Tensor, routing_index: int = 0) -> torch.Tensor:
        return self.experts[routing_index](hidden_states)


def _determine_routing_index(input_ids: torch.LongTensor, config: Qwen2PleConfig) -> int:
    """Determine the routing index from the control tokens in input_ids.

    Routing logic:
    1. If neither think_token_id nor no_think_token_id is configured -> return default_routing_index
    2. Scan input_ids for the last position of each control token
    3. The control token occurring last wins (injection defence: a control token from the system
       template normally comes after the user input)
    4. If neither is found -> return default_routing_index

    Args:
        input_ids: input token sequence, shape [batch_size, seq_len]
        config: PLE config

    Returns:
        int: 0 (no_think / expert 0) or 1 (think / expert 1)
    """
    think_id = config.think_token_id
    no_think_id = config.no_think_token_id
    default_idx = getattr(config, "default_routing_index", 0)

    # No control token configured -> return the default
    if think_id is None and no_think_id is None:
        return default_idx

    # Find the last position of each control token across the whole batch
    # (all sequences in the batch are assumed to share the same route)
    last_think_pos = -1
    last_no_think_pos = -1

    if think_id is not None:
        think_mask = (input_ids == think_id)
        if think_mask.any():
            # Position of the last think token (index taken from the flattened nonzero result)
            last_think_pos = think_mask.nonzero()[-1, -1].item()

    if no_think_id is not None:
        no_think_mask = (input_ids == no_think_id)
        if no_think_mask.any():
            last_no_think_pos = no_think_mask.nonzero()[-1, -1].item()

    # Neither control token found in input_ids -> use the default
    if last_think_pos == -1 and last_no_think_pos == -1:
        return default_idx

    # The control token occurring last wins.
    # Safety rationale: if a user injects /think in a message but the system template places
    # /no_think afterwards, the later /no_think overrides the injection.
    if last_think_pos > last_no_think_pos:
        return 1  # think mode
    else:
        return 0  # no_think mode


class Qwen2PleDecoderLayer(Qwen2DecoderLayer):
    """Decoder layer: attention is identical to Qwen2, the MLP is replaced by Qwen2PleMLP, and forward passes routing_index through."""

    def __init__(self, config: Qwen2PleConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2PleMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        routing_index: int = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, routing_index=routing_index)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen2PleModel(Qwen2Model):
    """Backbone model: determines routing_index at the start of forward and passes it to every layer's MLP.

    Sources of routing_index (by priority):
    1. routing_index passed explicitly in kwargs (cached by prepare_inputs_for_generation during generate, or set by the user)
    2. Computed from the control tokens in input_ids (training)
    3. Fallback to config.default_routing_index (default = 0 / no_think)
    """

    def __init__(self, config: Qwen2PleConfig):
        super().__init__(config)
        # Replace the Qwen2DecoderLayer created by the parent class with Qwen2PleDecoderLayer (two-expert MLP + routing_index)
        self.layers = nn.ModuleList(
            [Qwen2PleDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        # ===== Routing decision =====
        # Priority 1: routing_index taken from kwargs
        #   - generate: computed and cached on the first step by Qwen2PleForCausalLM.prepare_inputs_for_generation
        #   - the user can also pass routing_index=0/1 to force a route
        routing_index = kwargs.pop("routing_index", None)

        # Priority 2: computed from the control tokens in input_ids
        #   - training: every forward reaches this, since training does not go through prepare_inputs_for_generation
        #   - first generate step: also reached if prepare_inputs_for_generation did not set it
        if routing_index is None and input_ids is not None:
            routing_index = _determine_routing_index(input_ids, self.config)

        # Priority 3: fallback default
        #   - only when input_ids is None (only inputs_embeds given) and routing_index is not specified
        if routing_index is None:
            routing_index = getattr(self.config, "default_routing_index", 0)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                routing_index=routing_index,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Qwen2PleForCausalLM(Qwen2ForCausalLM):
    """Top-level causal LM using Qwen2PleModel as the backbone.

    Overrides prepare_inputs_for_generation to pass the routing decision correctly during generate:
    - if the user sets a route via generate(routing_index=N), that value is always used
    - otherwise it is detected from the control tokens in input_ids
    - the result is cached in self._cached_routing_index for debugging/testing
    """

    def __init__(self, config: Qwen2PleConfig):
        super().__init__(config)
        self.model = Qwen2PleModel(config)
        # Cache the routing_index during generate so later steps do not lose the route
        self._cached_routing_index = None

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        routing_index=None,
        **kwargs,
    ):
        """Overridden to pass the routing decision correctly during generate.

        Routing priority:
        1. Passed explicitly via generate(routing_index=N) -> used directly, no auto-detection
        2. Not passed (routing_index=None) -> detected from the control tokens in input_ids
        3. Auto-detection finds nothing -> config.default_routing_index

        Key fact: in generate's autoregressive loop, the input_ids passed to this method are always
        the **full accumulated sequence** (original prompt + generated tokens), not just the new token.
        Trimming happens inside the parent method (cache_dependant_input_preparation).

        Note: the routing_index argument keeps being passed through model_kwargs in the generate loop,
        so a value passed on the first step is retained for all later steps.
        """
        # Routing decision: prefer the value passed explicitly by the user, otherwise auto-detect
        if routing_index is not None:
            # Set explicitly via generate(routing_index=N)
            self._cached_routing_index = routing_index
        else:
            # Auto-detect the control token from the full input_ids
            self._cached_routing_index = _determine_routing_index(input_ids, self.config)

        # Call the parent prepare_inputs_for_generation to get the standard model_inputs
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        # Inject routing_index so it reaches the kwargs of Qwen2PleModel.forward
        model_inputs["routing_index"] = self._cached_routing_index
        return model_inputs


__all__ = [
    "Qwen2PleExpertMLP",
    "Qwen2PleMLP",
    "Qwen2PleDecoderLayer",
    "Qwen2PleModel",
    "Qwen2PleForCausalLM",
]
