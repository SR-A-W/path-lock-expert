# coding=utf-8
"""Path-Lock Expert (PLE) model implementations.

Each submodule provides a standalone PLE variant for one backbone family:

- ``qwen3_ple``      : Qwen3 family (used for the main results in the paper)
- ``qwen2_ple``      : Qwen2.5 family
- ``llama_ple``      : Llama-3.1 family
- ``phi4_mini_ple``  : Phi-4-mini family

All variants share the same design: every decoder layer carries two expert
MLPs (expert 0 = /no_think, expert 1 = /think) behind shared attention,
embeddings, normalization, and LM head. Routing is deterministic and
sequence-level, driven by the control token found in the prompt.

Backward compatibility: checkpoints released under the legacy ``*_pl_moe``
``model_type`` are aliased below so that ``AutoConfig``/``AutoModelForCausalLM``
continue to resolve them to the renamed classes. Checkpoints published with
``trust_remote_code=True`` are self-contained and unaffected either way.
"""

try:
    from transformers import AutoConfig, AutoModelForCausalLM

    from .qwen2_ple import Qwen2PleConfig, Qwen2PleForCausalLM
    from .llama_ple import LlamaPleConfig, LlamaPleForCausalLM

    _LEGACY_ALIASES = [
        ("qwen2_pl_moe", Qwen2PleConfig, Qwen2PleForCausalLM),
        ("llama_pl_moe", LlamaPleConfig, LlamaPleForCausalLM),
    ]
    for _legacy_type, _cfg_cls, _model_cls in _LEGACY_ALIASES:
        _compat_cfg = type(
            _cfg_cls.__name__ + "Legacy", (_cfg_cls,), {"model_type": _legacy_type}
        )
        try:
            AutoConfig.register(_legacy_type, _compat_cfg)
            AutoModelForCausalLM.register(_compat_cfg, _model_cls)
        except ValueError:
            pass  # already registered in this process
except ImportError:
    # transformers not installed; submodules can still be imported directly.
    pass
