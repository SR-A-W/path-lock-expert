#!/bin/bash
#SBATCH --job-name=eval_qwen3_4b_sup27k
#SBATCH --partition=your_partition
#SBATCH --account=your_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=60:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x/%x_%A_%a.out
#SBATCH --error=logs/%x/%x_%A_%a.err

# ========= Qwen3-4B PLE Superior 27k+27k 评估 =========

mkdir -p "logs/${SLURM_JOB_NAME}"

_SAVED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
_SAVED_SLURM_JOB_GPUS="${SLURM_JOB_GPUS}"
_SAVED_GPU_DEVICE_ORDINAL="${GPU_DEVICE_ORDINAL}"

# NOTE:----------训练完成后更新 MODEL_PATH----------
MODEL_PATH="${PROJECT_ROOT}/train/saves/qwen3_4b_superior_27k_27k"
EVAL_RESULTS_BASE="${PROJECT_ROOT}/eval/results"
CATEGORY="ple"
MODEL_NAME="qwen3_4b_ple_superior_27k_27k"

source ~/.bashrc
conda activate evalscope

if [ -n "${_SAVED_CUDA_VISIBLE_DEVICES}" ] && [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
  export CUDA_VISIBLE_DEVICES="${_SAVED_CUDA_VISIBLE_DEVICES}"
fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
  if [ -n "${_SAVED_SLURM_JOB_GPUS}" ]; then export CUDA_VISIBLE_DEVICES="${_SAVED_SLURM_JOB_GPUS}";
  elif [ -n "${_SAVED_GPU_DEVICE_ORDINAL}" ]; then export CUDA_VISIBLE_DEVICES="${_SAVED_GPU_DEVICE_ORDINAL}"; fi
fi

MAX_GPU_RETRIES=6; GPU_RETRY_INTERVAL=10; GPU_FOUND=false
for retry in $(seq 1 ${MAX_GPU_RETRIES}); do
  if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_FOUND=true; echo "PyTorch CUDA check passed on attempt ${retry}."; break; fi
  echo "WARNING: waiting for CUDA (attempt ${retry}/${MAX_GPU_RETRIES})..."; sleep ${GPU_RETRY_INTERVAL}
done
if [ "${GPU_FOUND}" = false ]; then echo "ERROR: no CUDA"; exit 1; fi

cd "${PROJECT_ROOT}"

TASKS=(math500 aime24 mmlu_stem gpqa_diamond)
MODES=(think no_think)

a=${SLURM_ARRAY_TASK_ID:-0}
task_idx=$(( a / 2 )); mode_idx=$(( a % 2 ))
TASK="${TASKS[$task_idx]}"; MODE="${MODES[$mode_idx]}"

RESULT_DIR="${EVAL_RESULTS_BASE}/${CATEGORY}/${MODEL_NAME}/${TASK}/${MODE}"
mkdir -p "$RESULT_DIR"

echo "[${TASK^^}] evaluating ${MODE}"
if [ ! -d "${MODEL_PATH}" ]; then echo "ERROR: Model path does not exist: ${MODEL_PATH}"; exit 1; fi

if [ "$TASK" == "math500" ]; then
  DATASET_NAME="math_500"
  PROMPT_TEMPLATE="{question}\nSolve step by step. Put your final answer in \\\\boxed{{}}./${MODE}"
  DATASET_ARGS="{\"${DATASET_NAME}\": {\"prompt_template\": \"${PROMPT_TEMPLATE}\", \"aggregation\": \"mean_and_pass_at_k\"}}"
elif [ "$TASK" == "aime24" ]; then
  DATASET_NAME="aime24"
  PROMPT_TEMPLATE="{question}\nSolve step by step. Put your final answer in \\\\boxed{{}}./${MODE}"
  DATASET_ARGS="{\"${DATASET_NAME}\": {\"prompt_template\": \"${PROMPT_TEMPLATE}\", \"aggregation\": \"mean_and_pass_at_k\"}}"
elif [ "$TASK" == "mmlu_stem" ]; then
  DATASET_NAME="mmlu"
  PROMPT_TEMPLATE="Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.\n{question}\n{choices}\n/${MODE}"
  DATASET_ARGS="{\"${DATASET_NAME}\": {\"subset_list\": [\"abstract_algebra\", \"astronomy\", \"college_biology\", \"college_chemistry\", \"college_computer_science\", \"college_mathematics\", \"college_physics\", \"computer_security\", \"conceptual_physics\", \"electrical_engineering\", \"elementary_mathematics\", \"high_school_biology\", \"high_school_chemistry\", \"high_school_computer_science\", \"high_school_mathematics\", \"high_school_physics\", \"high_school_statistics\", \"machine_learning\"], \"prompt_template\": \"${PROMPT_TEMPLATE}\", \"few_shot_num\": 0}}"
elif [ "$TASK" == "gpqa_diamond" ]; then
  DATASET_NAME="gpqa_diamond"
  PROMPT_TEMPLATE="Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.\n{question}\n{choices}\n/${MODE}"
  DATASET_ARGS="{\"${DATASET_NAME}\": {\"prompt_template\": \"${PROMPT_TEMPLATE}\"}}"
fi

evalscope eval \
  --model "${MODEL_PATH}" \
  --datasets "${DATASET_NAME}" \
  --repeats 5 \
  --dataset-args "${DATASET_ARGS}" \
  --work-dir "${RESULT_DIR}" \
  --generation-config '{"max_tokens": 32768}'

echo "Evaluation finished at: $(date)"
