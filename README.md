# Path-Lock Expert: Separating Reasoning Mode in Hybrid Thinking via Architecture-Level Separation

<p align="left">
  <a href="https://arxiv.org/abs/2604.27201"><img src="https://img.shields.io/badge/arXiv-2604.27201-b31b1b.svg" alt="arXiv"></a>
  <a href="https://colmweb.org/AcceptedPapers.html"><img src="https://img.shields.io/badge/COLM%202026-Accepted-2e7d32.svg" alt="COLM 2026"></a>
  <a href="https://openreview.net/forum?id=vifGxn9AUq"><img src="https://img.shields.io/badge/OpenReview-vifGxn9AUq-8c1d40.svg" alt="OpenReview"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://github.com/SR-A-W/agent-team-work-zone"><img src="https://img.shields.io/badge/Developed%20with-AT%20WorkZone-6f42c1.svg" alt="Developed with ATWZ"></a>
</p>

Official implementation of **Path-Lock Expert (PLE)** — an architecture-level solution to *reasoning leakage* in hybrid-thinking LLMs (models that expose both a `/think` mode — full Chain-of-Thought — and a `/no_think` mode — direct answer).

> **Paper**: [Path-Lock Expert: Separating Reasoning Mode in Hybrid Thinking via Architecture-Level Separation](https://arxiv.org/abs/2604.27201) (COLM 2026)

## Overview

**Core idea.** Replace the single MLP in each decoder layer with two semantically locked experts — expert 0 for `/no_think`, expert 1 for `/think` — while keeping attention, embeddings, normalization, and the LM head shared. Routing is fully deterministic and sequence-level, selected by the control token in the prompt. No learnable router, no soft routing, no routing loss.

**Why this works.** In hybrid-thinking SFT, the two modes compete for the same MLP capacity. Under standard training, the `/no_think` mode leaks CoT-style output and inflates length. PLE gives each mode a dedicated parameter subspace, so SFT signals from each mode no longer overwrite one another.

**Control-token detection.** Two mechanisms are provided, matching the two backbone situations:

- `qwen2_ple` (Qwen2.5 family): `/think` and `/no_think` are registered as dedicated tokenizer tokens (`src/utils/register_control_tokens.py`), and routing keys on the token IDs.
- `qwen3_ple` (Qwen3 family, used for the main results): no tokenizer changes — the router detects the control words at the decoded-string level with a guarded matcher (anchor-bounded scan, last occurrence wins), so it works with backbones whose tokenizers cannot register the control words as atomic tokens.

## Model Zoo

The main PLE checkpoint (`Qwen3-4B-PLE`, backbone weights initialized from Qwen3-4B-Instruct-2507) is being released on the Hugging Face Hub; the link will be added here once the upload is finalized.

Legacy checkpoints published under the older `*_pl_moe` naming remain loadable: they are self-contained (`trust_remote_code=True`), and this repo additionally registers `AutoConfig`/`AutoModelForCausalLM` aliases mapping the legacy `model_type` strings to the renamed classes (see `src/models/__init__.py`).

## Quick Start

### 1. Environment

Two conda environments are used (names below are suggestions):

```bash
conda create -n llms python=3.10
conda activate llms
pip install torch==2.4.0 deepspeed accelerate
# Install the training framework — see train/LLaMA-Factory/README.md
# Install the transformers fork (optional for inference) — see src/transformers/README.md
```

```bash
conda create -n evalscope python=3.10
conda activate evalscope
pip install evalscope vllm
# Install SkyThought — see eval/SkyThought/README.md
```

### 2. Download source models

```bash
python -m src.utils.download_source_models
```

Source models land under `./models/source/` (Qwen2.5-7B-Instruct, Qwen3-4B Base/Instruct, LLaMA-3.1-8B, Phi-4-mini).

### 3. Initialize PLE weights from a source model

```bash
python -m src.utils.copy_weights_to_ple \
  --source_model_path ./models/source/Qwen3-4B-Instruct-2507 \
  --output_path ./models/qwen3-4b-PLE-initialized
```

The initialization step duplicates the base-model MLP into 2 experts. For the Qwen2.5 family, additionally register the control tokens:

```bash
python -m src.utils.register_control_tokens \
  --model_path ./models/qwen2.5-7b-PLE-initialized
```

(The Qwen3 PLE variant needs no tokenizer changes — routing detects the control words at the string level.)

### 4. Train

Main result (Qwen3-4B, PLE, superior-reasoning corpus):

```bash
llamafactory-cli train train/tasks/qwen3_4b_superior_27k_27k.yaml
```

Naive-mix 140k baseline (Qwen2.5-7B):

```bash
llamafactory-cli train train/tasks/qwen_140k.yaml
```

See `slurm_examples/` for representative SLURM launch scripts. The full set of training configs is under `train/tasks/`.

### 5. Evaluate

```bash
sbatch slurm_examples/eval_superior_27k_27k.slurm
```

Benchmarks used in the paper: **MATH500**, **AIME24**, **MMLU-STEM**, **GPQA-Diamond**. Evaluation is driven by `evalscope` with the prompt format `"{question}\n...\n/${MODE}"` where `${MODE} ∈ {think, no_think}`.

## Repository Structure

```
.
├── README.md                   ← this file
├── DEVELOPMENT.md              ← developer guide (architecture, conventions)
├── LICENSE                     ← Apache 2.0
│
├── src/
│   ├── models/                 ← PLE model code (standalone, absolute imports)
│   │   ├── qwen3_ple/          ← Qwen3 family (main results)
│   │   ├── qwen2_ple/          ← Qwen2.5 family
│   │   ├── llama_ple/          ← Llama-3.1 family
│   │   └── phi4_mini_ple/      ← Phi-4-mini family
│   ├── datasets/               ← dataset generation / filtering pipeline
│   ├── utils/                  ← weight copying, control-token registration
│   └── tests/                  ← pytest suite (architecture, routing, forward-pass)
│
├── train/
│   ├── LLaMA-Factory/          ← submodule placeholder (see README inside)
│   ├── tasks/                  ← training YAML configs
│   └── data/                   ← training data (regenerate via src/datasets/)
│
├── eval/
│   ├── scripts/                ← evaluation post-processing, table/plot generation
│   └── results_summary.csv     ← aggregated evaluation results
│
├── models/                     ← model weights (see models/README.md)
└── slurm_examples/             ← representative SLURM scripts
```

### Submodules

Three external frameworks are used but not vendored. Each directory contains a `README.md` with instructions for restoring from the official upstream:

| Submodule | Official upstream | Purpose |
|-----------|-------------------|---------|
| `src/transformers/` | `huggingface/transformers` v4.56.2 | Modular model development (optional — production code in `src/models/` is standalone) |
| `train/LLaMA-Factory/` | `hiyouga/LLaMA-Factory` | Training framework |
| `eval/SkyThought/` | `NovaSky-AI/SkyThought` | Reasoning-benchmark eval framework |

## Reproducing Tables / Figures

| Paper element | Script |
|---------------|--------|
| Main results (accuracy + length) table | `eval/scripts/generate_results_xlsx.py` |
| Leakage table | `eval/scripts/tables/tab_leakage.py` |
| Main-result accuracy/length plots | `eval/scripts/generate_plots_main_results.py` |
| Backbone ablation plots | `eval/scripts/generate_plots_ablation_base.py` |
| Dataset ablation plots | `eval/scripts/generate_plots_ablation_dataset.py` |
| Average output length stats | `eval/scripts/avg_output_length_evalscope.py` |
| Reflective-token stats | `eval/scripts/reflective_token_stats_evalscope.py` |

## Citation

```bibtex
@article{wang2026pathlock,
  title={Path-Lock Expert: Separating Reasoning Mode in Hybrid Thinking via Architecture-Level Separation},
  author={Wang, Shouren and Yang, Wang and Ma, Chuang and Ganguly, Debargha and Singh, Vikash and Song, Chaoda and Li, Xinpeng and Long, Xianxuan and Chaudhary, Vipin and Han, Xiaotian},
  journal={arXiv preprint arXiv:2604.27201},
  year={2026},
  note={Accepted to COLM 2026}
}
```

## License

Apache 2.0. See `LICENSE`.
