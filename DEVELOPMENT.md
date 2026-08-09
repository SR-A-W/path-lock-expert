# Developer Guide

This file provides technical guidance for extending or debugging the codebase.

## Project Overview

Research project implementing **Path-Lock Expert (PLE)** to address reasoning leakage in hybrid-thinking LLMs. The problem: models in `no_think` mode still produce Chain-of-Thought reasoning and excessively long outputs.

The solution: a MoE architecture with 2 expert MLPs per layer (expert 0 = no_think, expert 1 = think), shared attention layers, and deterministic sequence-level hard routing based on control tokens (`/think`, `/no_think`). No learnable router — routing is purely token-driven.

Base models: Qwen2.5-7B-Instruct, Qwen3-4B (Base/Instruct variants), LLaMA-3.1-8B, Phi-4-mini. Both experts are initialized from the same base model weights and differentiated via SFT.

## Environment & Commands

```bash
conda activate llms        # Training environment (LLaMA-Factory, DeepSpeed, transformers fork)
conda activate evalscope   # Evaluation environment (evalscope, vllm, SkyThought)
```

### Testing
```bash
# Unit tests (config, routing, basic functionality)
python -m pytest src/tests/test_ple_basic.py -v -s

# Static weight consistency tests (no model loading)
python -m pytest src/tests/test_ple_weights_lite.py -v -s

# Routing & forward pass tests (4-bit quantized, needs GPU)
python -m pytest src/tests/test_ple_weights.py -v -s
```

### Model Development Workflow
```bash
# 1. Edit the modular file (source of truth)
#    src/transformers/src/transformers/models/qwen2_ple/modular_qwen2_ple.py
#    (Note: src/transformers/ is a transformers fork; see src/transformers/README.md)

# 2. Generate modeling file from modular
cd src/transformers
python utils/modular_model_converter.py \
  src/transformers/models/qwen2_ple/modular_qwen2_ple.py

# 3. Copy to production location and fix imports
#    Copy from src/transformers/models/qwen2_ple/ → src/models/qwen2_ple/
#    Change relative imports (from ...activations) → absolute (from transformers.activations)
```

### Weight Initialization
```bash
python -m src.utils.copy_weights_to_ple \
  --source_model_path ./models/source/Qwen2.5-7B-Instruct \
  --output_path ./models/ple_initialized
```

### Training (LLaMA-Factory)
```bash
cd train/LLaMA-Factory
llamafactory-cli train ../tasks/qwen_140k.yaml
```

### Evaluation (SLURM)
See `slurm_examples/` for representative SLURM scripts covering the full pipeline.
Benchmarks: MATH500, AIME24, MMLU-STEM, GPQA-Diamond.

## Architecture

### Dual-Location Pattern
- **Development** (not vendored; restore from upstream, see `src/transformers/README.md`): `src/transformers/` — a transformers 4.56.2 fork where PLE modular files live. Edit `modular_*.py`, generate `modeling_*.py` via `utils/modular_model_converter.py`.
- **Production**: `src/models/<base>_ple/` — standalone copy with absolute imports, ready to run without the fork.

### Key Model Classes (in `modeling_qwen2_ple.py`)
- `Qwen2PleConfig` — extends `Qwen2Config` with routing params (`think_token_id`, `no_think_token_id`, `default_routing_index`)
- `Qwen2PleExpertMLP` — single MLP expert (identical structure to `Qwen2MLP`)
- `Qwen2PleMLP` — container for 2 experts with routing dispatch
- `Qwen2PleDecoderLayer` — decoder layer with dual-expert MLP
- `Qwen2PleModel` — main model, detects control tokens in `input_ids` to set routing
- `Qwen2PleForCausalLM` — causal LM wrapper, handles `prepare_inputs_for_generation`

The same pattern is repeated for `qwen3_ple`, `llama_ple`, and `phi4_mini_ple`.

### Routing Priority
1. Explicit `routing_index` parameter (highest priority).
2. Control tokens in `input_ids` (`/think` → 1, `/no_think` → 0).
3. `config.default_routing_index` (fallback, default = 0 i.e. no_think).

### Control Tokens
- `/think`: token ID 151665 → routes to expert 1.
- `/no_think`: token ID 151666 → routes to expert 0.
- Registered via `src/utils/register_control_tokens.py` (token ID values may differ per base tokenizer; the script ensures the two IDs are assigned consistently).

## Key Directories
- `src/models/` — production model code (standalone, absolute imports).
- `src/transformers/` — submodule placeholder (see README inside).
- `src/utils/` — weight copying, token registration scripts.
- `src/tests/` — pytest test suite.
- `train/LLaMA-Factory/` — submodule placeholder (see README inside).
- `train/tasks/` — training YAML configs.
- `eval/scripts/` — evaluation post-processing and table/plot generation.
- `eval/SkyThought/` — submodule placeholder (see README inside).
- `slurm_examples/` — representative SLURM scripts for the full pipeline.
- `plots/` — rendered figures used in the paper.

## Conventions
- Documentation is mixed Chinese and English; code comments are likewise mixed.
- Prefer detailed comments in non-trivial code paths.
- `TODO.md` checklists in relevant subdirectories track implementation progress for active refactors.
