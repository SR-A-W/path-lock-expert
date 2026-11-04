# coding=utf-8
# Copyright 2026 Hybrid-Expert-Thinking Project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""PyTorch Qwen3 PLE (Path-Lock Expert) model.

Sequence-level hard routing: the control word `/think` or `/no_think` at the end of the user
message decides whether the whole sequence uses expert 0 or expert 1.

Routing detection rules:
1. **Anchor**: find the last `<|im_start|>assistant` token pair (special tokens 151644 + 77091);
   it is the right boundary of the scan region and cleanly separates the prompt from the response.
2. **scan_region**: the tokens before the anchor (branch 1); if there is no anchor (bare inference
   without a chat template, branch 2), the whole sequence is scanned.
3. **decode + last-wins**: after tokenizer.decode, run regex `(?<!<)/no_think(?!\w)|(?<!<)/think(?!\w)`
   findall and take the **last** match. `(?<!<)` excludes the closing tag `</think>`; `(?!\w)`
   avoids substrings like `/think_hard`.
4. **NO_MATCH behaviour**: controlled by `config.strict_routing_match`:
   - `True` (default) -> raise (fail-fast)
   - `False` -> log a warning and fall back to `config.default_routing_index`

`_determine_routing_index` returns `LongTensor[B]`, and the MLP / DecoderLayer take `LongTensor[B]`,
so each sample in a batch can have its own route.
"""

import logging as py_logging
import re
from typing import Optional

import torch
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.generic import check_model_inputs
from ..qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3Model,
)
from .configuration_qwen3_ple import Qwen3PLEConfig


logger = py_logging.getLogger(__name__)


# Anchor token pair: <|im_start|> + assistant.
# These two special tokens do not occur consecutively in naturally encoded text
# (unless a user injects them literally).
_ANCHOR_BIGRAM = (151644, 77091)

# Core regex of routing detection.
# - alternation order: /no_think comes first (the strings overlap; /no_think contains /think)
# - (?<!<): negative lookbehind, excludes the closing tag </think>
# - (?!\w): negative lookahead, avoids false hits on substrings such as /think_hard
_ROUTING_PATTERN = re.compile(r'(?<!<)/no_think(?!\w)|(?<!<)/think(?!\w)')


# Module-level tokenizer cache, keyed by model path.
# Avoids reloading the tokenizer on every routing detection; model instances share one tokenizer.
_TOKENIZER_CACHE: dict = {}


def _get_routing_tokenizer(config):
    """Lazily load the tokenizer, cached by config._name_or_path.

    Routing detection needs tokenizer.decode to turn the token sequence back into a string for
    regex matching. This returns the tokenizer that matches the model weights.
    """
    path = getattr(config, "_name_or_path", None) or getattr(config, "name_or_path", None)
    if path is None:
        raise ValueError(
            "Qwen3PLE routing detection requires a tokenizer, but config._name_or_path is not set. "
            "This is normally set automatically by from_pretrained(). "
            "If loading manually, pass tokenizer reference via prepare_inputs_for_generation "
            "or call from_pretrained with model directory."
        )
    if path not in _TOKENIZER_CACHE:
        from ..auto import AutoTokenizer
        _TOKENIZER_CACHE[path] = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return _TOKENIZER_CACHE[path]


def _find_last_bigram(ids_row: torch.LongTensor, anchor=_ANCHOR_BIGRAM) -> Optional[int]:
    """Find the start position of the last occurrence of the `anchor` token pair.

    Args:
        ids_row: 1D LongTensor (a single sequence)
        anchor: tuple (t0, t1) of length 2

    Returns:
        Optional[int]: start position i with ids_row[i]==t0 and ids_row[i+1]==t1; None if not found
    """
    if ids_row.shape[0] < 2:
        return None
    t0, t1 = anchor
    # Vectorised: matches[i] = (ids[i]==t0 AND ids[i+1]==t1)
    matches = (ids_row[:-1] == t0) & (ids_row[1:] == t1)
    if not matches.any():
        return None
    return int(matches.nonzero()[-1, 0].item())


def _detect_routing_for_row(
    ids_row: torch.LongTensor, config: Qwen3PLEConfig, tokenizer
) -> int:
    """Per-row routing detection (anchor + regex + last-wins).

    Args:
        ids_row: 1D LongTensor, a single sequence
        config: Qwen3PLEConfig
        tokenizer: HuggingFace tokenizer (used for decode)

    Returns:
        int: 0 (no_think / expert 0) or 1 (think / expert 1)
    """
    default_idx = getattr(config, "default_routing_index", 0)
    strict = getattr(config, "strict_routing_match", True)

    # Step 1: anchor (right boundary)
    a_pos = _find_last_bigram(ids_row, _ANCHOR_BIGRAM)

    # Step 2: choose scan_region by branch
    if a_pos is not None:
        # Branch 1: training / inference with a chat template. Everything before the anchor is the
        # prompt region (the system and user turns of the chat template); the response is
        # structurally excluded.
        scan_region = ids_row[:a_pos]
    else:
        # Branch 2: bare inference without a template. The whole sequence is the user input (normally no response at inference).
        scan_region = ids_row

    # Step 3: decode + regex, last match wins
    if scan_region.shape[0] == 0:
        s = ""
    else:
        s = tokenizer.decode(scan_region.tolist())
    matches = _ROUTING_PATTERN.findall(s)

    # Step 4: NO_MATCH behaviour
    if not matches:
        if strict:
            raise ValueError(
                f"Qwen3PLE routing: control token /think or /no_think not found in input "
                f"(scan_region len={int(scan_region.shape[0])}, has_anchor={a_pos is not None}). "
                f"Either add control word to input end, or set config.strict_routing_match=False "
                f"to fall back to default_routing_index={default_idx}."
            )
        logger.warning(
            "Qwen3PLE routing NO_MATCH; falling back to config.default_routing_index=%d. "
            "Set config.strict_routing_match=True to raise instead.",
            default_idx,
        )
        return default_idx

    last_match = matches[-1]
    # last_match is the string "/think" or "/no_think"
    return 0 if last_match == "/no_think" else 1


def _determine_routing_index(
    input_ids: torch.LongTensor, config: Qwen3PLEConfig, tokenizer=None
) -> torch.LongTensor:
    """Per-sample routing detection. Returns `LongTensor[B]`; works for any batch size B.

    Args:
        input_ids: 2D LongTensor [B, S]
        config: Qwen3PLEConfig
        tokenizer: optional; lazily loaded if None

    Returns:
        LongTensor[B]: routing index (0 or 1) of each sequence
    """
    if tokenizer is None:
        tokenizer = _get_routing_tokenizer(config)
    B = input_ids.shape[0]
    results = [_detect_routing_for_row(input_ids[b], config, tokenizer) for b in range(B)]
    return torch.tensor(results, dtype=torch.long, device=input_ids.device)


class Qwen3PLEExpertMLP(Qwen3MLP):
    """A single MLP expert (identical in structure to Qwen3MLP), used in the PLE two-expert container.

    Qwen3MLP inherits from GemmaMLP and has gate_proj, up_proj and down_proj, all without bias.
    forward: down_proj(act_fn(gate_proj(x)) * up_proj(x))
    """

    pass


class Qwen3PLEMLP(nn.Module):
    """Path-Lock two-expert MLP container (Scheme A for training / Scheme B for inference,
    switched automatically by the gradient context).

    Holds two independent MLP experts:
    - experts[0]: no_think expert
    - experts[1]: think expert

    **forward picks the dispatch scheme from `torch.is_grad_enabled()`**:

    **Scheme A (grad enabled = training / backward possible)**: **both** experts are computed and a
    per-sample mask selects `out = out0*(1-mask) + out1*mask`. Costs 2x MLP FLOPs, but gives:
    - correct per-sample gradients (think/no_think can be mixed within a batch)
    - **structural ZeRO-2 safety**: both experts always take part in forward, so no expert ever has
      a zero-gradient step (benign for homogeneous batches and for multi-GPU reduce)

    **Scheme B (grad disabled = inference / no backward)**: only the routed expert is computed,
    ~1x MLP FLOPs. Without a backward pass there are no gradients, so the zero-gradient problem
    cannot occur, and half the MLP compute is saved.
    - homogeneous batch (common at inference, all samples share one route) -> a single expert call
    - mixed batch -> split by route, each segment goes only through its expert

    **Numerical equivalence**: for the selected expert, Scheme B is bit-identical to Scheme A
    (in A, `out_sel*1.0 + out_other*0.0` == out_sel); switching changes compute, not results.
    The switch uses `torch.is_grad_enabled()` rather than `self.training`: grad on = backward
    possible = must use A; grad off = pure forward = B is safe. This is more robust than
    model.training (it is not fooled by eval-time loss computation).
    """

    def __init__(self, config: Qwen3PLEConfig):
        super().__init__()
        self.experts = nn.ModuleList([Qwen3PLEExpertMLP(config) for _ in range(2)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_index: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, S, H]
            routing_index: [B] LongTensor, 0=no_think (expert 0), 1=think (expert 1)

        Returns:
            [B, S, H], output of the expert selected by routing_index for each sample
        """
        # ============================================================
        # Master switch: training (backward pass) or inference (no backward)?
        #   torch.is_grad_enabled()==True  -> gradients are computed (training) -> Scheme A
        #   torch.is_grad_enabled()==False -> no gradients (inference / no_grad) -> Scheme B
        #   Used instead of self.training: grad on = backward possible = must use A; grad off = pure forward = B is safe
        # ============================================================
        if torch.is_grad_enabled():
            # ========================================================
            # Scheme A (training): compute both experts, then pick per sample with a mask.
            #   Cost: 2x MLP compute (both experts run).
            #   Benefit: both experts take part in forward -> both receive real gradients in backward ->
            #            no step where an expert has zero gradient (ZeRO-2 safe).
            # ========================================================
            out0 = self.experts[0](hidden_states)               # A1: expert0 (no_think) on the whole batch -> [B,S,H]
            out1 = self.experts[1](hidden_states)               # A2: expert1 (think) on the whole batch -> [B,S,H]
            # A3: reshape routing_index from [B] to [B,1,1] so it broadcasts against [B,S,H].
            #   e.g. routing=[0,1] -> mask=[[[0.]],[[1.]]] (dtype matched to out0 to avoid casts)
            mask = routing_index.view(-1, 1, 1).to(out0.dtype)
            # A4: per-sample linear selection:
            #   routing=0 -> out0*(1-0)+out1*0 = out0 (expert 0)
            #   routing=1 -> out0*(1-1)+out1*1 = out1 (expert 1)
            return out0 * (1.0 - mask) + out1 * mask

        # ============================================================
        # Scheme B (inference): compute only the selected expert, ~1x MLP compute.
        #   No backward -> no gradients -> the zero-gradient problem cannot occur, so both experts
        #   need not run. The selected expert's output is bit-identical to Scheme A (the other
        #   expert is multiplied by an exact 0 there).
        #
        #   There is deliberately no separate fast path for homogeneous batches: the segmented
        #   path below already covers that case (the other expert's selection is all False and it
        #   is skipped), and end-to-end generate showed no measurable gain from a dedicated branch.
        # ============================================================
        # B1: split by route; each expert only computes its own part (the other is skipped for a homogeneous batch)
        out = torch.empty_like(hidden_states)                   # empty output tensor with the same shape as the input
        for e in (0, 1):                                        # handle expert 0 and expert 1 in turn
            sel = routing_index == e                            # boolean mask: which samples go to expert e (e.g. e=0 -> [T,F,T])
            if sel.any():                                       # if any sample goes to this expert
                out[sel] = self.experts[e](hidden_states[sel])  # feed only those samples to expert e and write back in place
        return out                                              # each expert only computes its own part -> total compute ~1x


class Qwen3PLEDecoderLayer(Qwen3DecoderLayer):
    """Decoder layer: attention is identical to Qwen3 (including q_norm/k_norm), the MLP is replaced
    by Qwen3PLEMLP, and forward passes `routing_index: LongTensor[B]` through.
    """

    def __init__(self, config: Qwen3PLEConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        # Replace the Qwen3MLP created by the parent class with the PLE two-expert container
        self.mlp = Qwen3PLEMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        routing_index: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if routing_index is None:
            raise ValueError(
                "Qwen3PLEDecoderLayer.forward requires routing_index (LongTensor[B]); "
                "got None. This should be supplied by Qwen3PLEModel.forward."
            )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention (shared layer, with q_norm/k_norm)
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

        # Fully Connected (routed to the selected expert)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, routing_index=routing_index)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3PLEModel(Qwen3Model):
    """Backbone model: determines `routing_index: LongTensor[B]` at the start of forward and passes it to every layer.

    Sources of routing_index (by priority):
    1. Passed explicitly in kwargs (cached by prepare_inputs_for_generation during generate; the user may also pass it)
    2. Detected automatically from input_ids (training / first generate step)
    3. Fallback: torch.full([B], config.default_routing_index) (only when input_ids is None)
    """

    def __init__(self, config: Qwen3PLEConfig):
        super().__init__(config)
        # Replace the Qwen3DecoderLayer created by the parent class with Qwen3PLEDecoderLayer
        self.layers = nn.ModuleList(
            [Qwen3PLEDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
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
        # Priority 1: routing_index taken from kwargs (LongTensor[B] or int)
        routing_index = kwargs.pop("routing_index", None)

        # Priority 2: detected automatically from the control tokens in input_ids
        if routing_index is None and input_ids is not None:
            routing_index = _determine_routing_index(input_ids, self.config)
        elif routing_index is not None and not isinstance(routing_index, torch.Tensor):
            # Accept int / list / other non-tensor input and convert it to a LongTensor
            device = input_ids.device if input_ids is not None else (
                inputs_embeds.device if inputs_embeds is not None else "cpu"
            )
            if isinstance(routing_index, int):
                B = (input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0])
                routing_index = torch.full((B,), routing_index, dtype=torch.long, device=device)
            else:
                routing_index = torch.tensor(routing_index, dtype=torch.long, device=device)

        # Priority 3: fallback default (only when input_ids is None and nothing was specified)
        if routing_index is None:
            default_idx = getattr(self.config, "default_routing_index", 0)
            B = inputs_embeds.shape[0] if inputs_embeds is not None else 1
            device = inputs_embeds.device if inputs_embeds is not None else "cpu"
            routing_index = torch.full((B,), default_idx, dtype=torch.long, device=device)

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


class Qwen3PLEForCausalLM(Qwen3ForCausalLM):
    """Top-level causal LM using Qwen3PLEModel as the backbone.

    Overrides prepare_inputs_for_generation to pass the routing decision correctly during generate:
    - if the user sets a route via `generate(routing_index=N)` (int / list / tensor), that value is always used
    - otherwise it is detected from the control tokens in input_ids
    - the result is cached in `self._cached_routing_index` (LongTensor[B]) for debugging/testing
    """

    def __init__(self, config: Qwen3PLEConfig):
        super().__init__(config)
        self.model = Qwen3PLEModel(config)
        # Cache the routing decision during generate so prefill and later decode steps use the same route
        self._cached_routing_index: Optional[torch.LongTensor] = None

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
        1. Passed explicitly via `generate(routing_index=N)` (int / list / Tensor) -> converted to LongTensor[B]
        2. Not passed -> detected automatically from the full input_ids
        3. On NO_MATCH, config.strict_routing_match decides the behaviour

        Key point: in generate's autoregressive loop, the input_ids passed to this method are always
        the **full accumulated sequence** (original prompt + generated tokens). Because `routing_index`
        travels through model_kwargs and cache_dependant_input_preparation does not trim kwargs,
        **the value cached on the first step is retained for all steps**.
        """
        device = input_ids.device
        B = input_ids.shape[0]

        if routing_index is not None:
            # Specified explicitly by the user; convert to LongTensor[B].
            if isinstance(routing_index, int):
                self._cached_routing_index = torch.full(
                    (B,), routing_index, dtype=torch.long, device=device
                )
            elif isinstance(routing_index, torch.Tensor):
                self._cached_routing_index = routing_index.to(dtype=torch.long, device=device)
            else:
                self._cached_routing_index = torch.tensor(
                    routing_index, dtype=torch.long, device=device
                )
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

        # Inject routing_index so it reaches the kwargs of Qwen3PLEModel.forward
        model_inputs["routing_index"] = self._cached_routing_index
        return model_inputs


__all__ = [
    "Qwen3PLEExpertMLP",
    "Qwen3PLEMLP",
    "Qwen3PLEDecoderLayer",
    "Qwen3PLEModel",
    "Qwen3PLEForCausalLM",
]
