# coding=utf-8
# Copyright 2026 Hybrid-Expert-Thinking Project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Qwen3 PLE (Path-Lock Expert) model configuration.

Routing detection: anchor + decode + last-wins.
"""

from transformers.utils import logging
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


logger = logging.get_logger(__name__)


class Qwen3PLEConfig(Qwen3Config):
    r"""
    Path-Lock Expert config class, extending Qwen3Config.

    Routing detection design:
    - There are no `think_token_id` / `no_think_token_id`: detection is based on
      tokenizer.decode + regex string matching, not on registered single token ids.
    - `strict_routing_match` is a safe-mode switch controlling behaviour on NO_MATCH.
    - `default_routing_index` is the fallback in non-strict mode.

    Routing decision (implemented in the modeling file):
    1. Locate the last `<|im_start|>assistant` anchor (if any).
    2. scan_region = tokens before the anchor (the whole sequence if there is no anchor).
    3. tokenizer.decode(scan_region) -> string.
    4. regex `(?<!<)/no_think(?!\w)|(?<!<)/think(?!\w)` findall; take the last match.
       - negative lookbehind `(?<!<)` excludes the closing tag `</think>`
       - negative lookahead `(?!\w)` avoids false hits on substrings like `/think_hard`
    5. On NO_MATCH, `strict_routing_match` decides the behaviour.

    Args:
        strict_routing_match (`bool`, *optional*, defaults to `True`):
            Safe-mode switch. If `True`, NO_MATCH raises `ValueError` (fail-fast);
            if `False`, fall back to `default_routing_index` and log a warning (inference tolerance).
        default_routing_index (`int`, *optional*, defaults to 0):
            Default route used on NO_MATCH (only when `strict_routing_match=False`).
        **kwargs:
            Remaining arguments are passed to `Qwen3Config`.
    """

    model_type = "qwen3_ple"

    def __init__(
        self,
        strict_routing_match: bool = True,
        default_routing_index: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.strict_routing_match = strict_routing_match
        self.default_routing_index = default_routing_index


__all__ = ["Qwen3PLEConfig"]
