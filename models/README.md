# `models/` — Model Weights (not included)

Model weights are not bundled in this repository:

1. **Size**: the source base models (Qwen2.5-7B, Qwen3-4B, LLaMA-3.1-8B, Phi-4-mini) and their PLE-initialized variants total tens of gigabytes.
2. **Reproducibility**: the PLE initialization is deterministic given the source model, so the initialized checkpoints can be reproduced locally in a few minutes.

## Expected directory layout (after reconstruction)

```
models/
├── source/                                   # Base models downloaded from official HuggingFace
│   ├── Qwen2.5-7B-Instruct/
│   ├── Qwen3-4B-Base/
│   ├── Qwen3-4B-Instruct-2507/
│   ├── Qwen3-8B-Base/
│   ├── Meta-Llama-3.1-8B-Instruct/
│   └── Phi-4-mini-instruct/
├── qwen2.5-7b-ple-initialized/            # PLE-initialized checkpoints (local-only)
├── qwen3_4b_ple_initialized/
├── qwen3_4b_base_ple_initialized/
├── qwen3_4b_instruct_ple_initialized/
├── qwen3_4b_base_instruct_ple_initialized/
├── qwen3_8b_ple_initialized/
├── llama_ple_initialized/
└── phi4_mini_ple_initialized/
```

## Reconstruction procedure

### 1. Download the base models from HuggingFace

The base models used in the paper are all **publicly available** on the official organisation accounts; no private repositories are involved.

| Base model | HuggingFace repo |
|-----------|-----------------|
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` |
| Qwen3-4B-Base | `Qwen/Qwen3-4B-Base` |
| Qwen3-4B-Instruct-2507 | `Qwen/Qwen3-4B-Instruct-2507` |
| Qwen3-8B-Base | `Qwen/Qwen3-8B-Base` |
| LLaMA-3.1-8B-Instruct | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Phi-4-mini-instruct | `microsoft/Phi-4-mini-instruct` |

Download via the helper stub or directly with `huggingface-cli`:

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/source/Qwen2.5-7B-Instruct
# ... repeat for other base models
```

See `src/utils/download_source_models.py` for the helper stub.

### 2. Initialize PLE checkpoints from each base model

From the repository root:

```bash
# Qwen2.5-7B → PLE
python -m src.utils.copy_weights_to_ple \
  --source_model_path ./models/source/Qwen2.5-7B-Instruct \
  --output_path ./models/qwen2.5-7b-ple-initialized

# Register the /think and /no_think control tokens
python -m src.utils.register_control_tokens \
  --model_path ./models/qwen2.5-7b-ple-initialized
```

Repeat for each base model, setting `--source_model_path` and `--output_path` per the expected layout above. The initialization step duplicates the base-model MLP into 2 experts; the token registration step adds `/think` (id 151665) and `/no_think` (id 151666) to the tokenizer.

Each initialization run takes ~5 minutes on a single GPU or ~10 minutes on CPU. Total storage requirement after reconstruction: roughly 120 GB for all 8 variants.

### 3. Verify (optional)

Run the static weight-consistency tests:

```bash
python -m pytest src/tests/test_ple_weights_lite.py -v -s
```

## Notes

- For `Qwen3-4B-Base` → Instruct conversion (the Stage 1 variant referenced in some configs), follow `slurm_examples/stage1_instruct_example.slurm`.
- Evaluation checkpoints trained with any of the training YAML configs in `train/tasks/` land under `train/saves/<run_name>/`, not here.
