#!/bin/bash
#SBATCH --job-name=qwen3_4b_sup_27k
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=your_partition
#SBATCH --account=your_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=120:00:00

# ========== Qwen3-4B PLE Superior 27k+27k 训练 (H200) ==========

source ~/.bashrc
conda activate llms
cd "${PROJECT_ROOT}"

module load CUDA/12.8.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false
export FORCE_TORCHRUN=1
export OPTIM_TORCH=0

echo "Job started at: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CONFIG="train/tasks/qwen3_4b_superior_27k_27k.yaml"
echo "Training config: ${CONFIG}"
llamafactory-cli train "${CONFIG}"

echo "Training finished at: $(date)"
