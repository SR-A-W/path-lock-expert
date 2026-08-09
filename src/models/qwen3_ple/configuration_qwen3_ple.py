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
    Path-Lock Expert 配置类，继承自 Qwen3Config。

    路由检测设计要点：
    - 移除 `think_token_id` / `no_think_token_id`：新路由检测基于 tokenizer.decode + regex
      字符串匹配，不再依赖单 token id 注册
    - 新增 `strict_routing_match`：safe-mode 开关，控制 NO_MATCH 时的行为
    - `default_routing_index` 沿用，作为非严格模式下的 fallback

    路由决策（在 modeling 中实现）：
    1. 定位最后一个 `<|im_start|>assistant` 锚点（如果有）
    2. scan_region = 锚点之前的 token（无锚点则整段）
    3. tokenizer.decode(scan_region) → 字符串
    4. regex `(?<!<)/no_think(?!\w)|(?<!<)/think(?!\w)` findall，取最后一个匹配
       - negative lookbehind `(?<!<)` 排除 `</think>` 闭合标签
       - negative lookahead `(?!\w)` 防 `/think_hard` 类子串误触发
    5. NO_MATCH 时由 `strict_routing_match` 控制行为

    Args:
        strict_routing_match (`bool`, *optional*, defaults to `True`):
            safe-mode 开关。`True` 时 NO_MATCH 抛 `ValueError`（fail-fast，防 V2 灾难重演）；
            `False` 时退 `default_routing_index` 并 log warning（推理容错用）。
        default_routing_index (`int`, *optional*, defaults to 0):
            NO_MATCH 时（仅在 `strict_routing_match=False` 下）使用的默认路由。
        **kwargs:
            其余参数透传给 `Qwen3Config`。
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
