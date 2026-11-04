# coding=utf-8
# Copyright 2026 Hybrid-Expert-Thinking Project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Standalone Qwen3 PLE (Path-Lock Expert) module (imports use the pip-installed transformers).

Routing detection: anchor + tokenizer.decode + regex negative-lookbehind + last-occurrence-wins + safe-mode.
See the top-level docstring of modular_qwen3_ple.py for details.
"""

from .configuration_qwen3_ple import Qwen3PLEConfig
from .modeling_qwen3_ple import (
    Qwen3PLEDecoderLayer,
    Qwen3PLEExpertMLP,
    Qwen3PLEForCausalLM,
    Qwen3PLEMLP,
    Qwen3PLEModel,
    Qwen3PLEPreTrainedModel,
)

__all__ = [
    "Qwen3PLEConfig",
    "Qwen3PLEDecoderLayer",
    "Qwen3PLEExpertMLP",
    "Qwen3PLEForCausalLM",
    "Qwen3PLEMLP",
    "Qwen3PLEModel",
    "Qwen3PLEPreTrainedModel",
]
