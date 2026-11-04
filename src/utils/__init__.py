# -*- coding: utf-8 -*-
"""PLE utilities

Contents:
- copy_weights_to_ple: initialize a PLE checkpoint from a source model (duplicate the MLP into 2 experts)
- register_control_tokens: register /think and /no_think as tokenizer tokens (Qwen2.5 family)
- download_source_models: download source model weights
"""

from .download_source_models import download_model, verify_model

__all__ = [
    "download_model",
    "verify_model",
]
