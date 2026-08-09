# SLURM Example Scripts

This directory contains two representative SLURM scripts covering the main experiment reported in the paper: Qwen3-4B PLE trained with the curated "superior reasoning" 27k+27k corpus, and evaluated on the four reasoning benchmarks (MATH500, AIME24, MMLU-STEM, GPQA-Diamond).

Cluster-specific identifiers (partition, account, absolute paths) are replaced with placeholders; adapt the scripts to your environment with minor edits.

## Files

| File | Purpose |
|------|---------|
| `train_superior_27k_27k.slurm` | Stage 2 PLE SFT on the superior-reasoning 27k+27k corpus (2x H-class GPUs) |
| `eval_superior_27k_27k.slurm` | Full evaluation sweep: MATH500 / AIME24 / MMLU-STEM / GPQA-Diamond x {think, no_think} via an 8-way SLURM array |

## Adapting to Your Cluster

Before `sbatch`, edit or export the following:

| Placeholder | Replace with |
|-------------|--------------|
| `#SBATCH --partition=your_partition` | Your cluster partition name (e.g. `gpu`, `a100`, ...) |
| `#SBATCH --account=your_account` | Your account / billing code |
| `${PROJECT_ROOT}` | Absolute path of this repo on your cluster. Export it in your shell before submission, e.g.: `export PROJECT_ROOT=/path/to/anon-repo && sbatch slurm_examples/train_superior_27k_27k.slurm` |
| `conda activate llms` / `evalscope` | Your local conda env names (training / evaluation) |
| `module load CUDA/12.8.0` | Your cluster's CUDA module (CUDA 12.1+ should work) |

For the eval script, also edit `MODEL_PATH` near the top to point at the checkpoint produced by the training job.

## Environment Setup

The project uses two conda environments (names are suggestions; adjust to your setup):

- **Training env**: transformers 4.56.2 (see `src/transformers/README.md`) + LLaMA-Factory (see `train/LLaMA-Factory/README.md`) + `deepspeed`/`accelerate`.
- **Evaluation env**: `evalscope` + `vllm` + `SkyThought` (see `eval/SkyThought/README.md`).

## GPU Budget

- **Training** (`train_superior_27k_27k.slurm`): 2x H100/H200 class GPUs, ~120 h wall-clock.
- **Evaluation** (`eval_superior_27k_27k.slurm`): 1x A100/H-class GPU per array element, 8 array elements total, ~4 h per element.

## Other Configurations

The full set of training YAML configs is under `../train/tasks/` (60+ files covering all ablations: Qwen2.5-7B, Qwen3-4B Base/Instruct, Qwen3-8B, LLaMA-3.1-8B, Phi-4-mini across naive-mix / superior-reasoning / superior-hybrid dataset variants). Each can be launched with the training SLURM template by swapping the `CONFIG` line, e.g.:

```bash
CONFIG="train/tasks/qwen_140k.yaml"          
CONFIG="train/tasks/qwen3_4b_superior_65k_27k.yaml"   
CONFIG="train/tasks/qwen3_4b_base_instruct_stage1.yaml"   
```
