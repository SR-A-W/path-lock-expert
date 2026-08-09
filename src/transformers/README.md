# transformers (submodule placeholder)

This directory originally contains a **fork of HuggingFace Transformers (v4.56.2)** with added support for the Path-Lock Expert (PLE) model classes.

The submodule content is **not vendored in this repository**; restore it from the official upstream as described below.

## How to reproduce

If you need the modified transformers code to run training / inference, do one of the following:

1. **Use the official upstream** (minimal — PLE inference still works via `src/models/` standalone path):

   ```bash
   git clone --branch v4.56.2 https://github.com/huggingface/transformers.git .
   pip install -e .
   ```

   The PLE production code under `src/models/qwen2_ple/`, `src/models/qwen3_ple/`, `src/models/llama_ple/`, and `src/models/phi4_mini_ple/` is standalone (absolute imports from `transformers.*`) and does not require the fork.

2. **Rebuild the fork yourself**: apply the PLE modular files under `src/models/<model>_ple/` using the `modular_model_converter.py` workflow described in the top-level README. See `src/models/<model>_ple/modular_<model>_ple.py` for the source-of-truth modular definitions.

## What was in the fork

The fork adds 4 new model packages under `transformers/models/`:

- `qwen2_ple/`
- `qwen3_ple/`
- `llama_ple/`
- `phi4_mini_ple/`

Each package contains `modular_*.py` (source of truth) and `modeling_*.py` (generated via `utils/modular_model_converter.py`). All other files are identical to upstream v4.56.2.
