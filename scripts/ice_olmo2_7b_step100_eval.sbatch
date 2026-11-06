#!/bin/bash
#SBATCH --job-name=olmo2-7b-step100-eval
#SBATCH --account=ece
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
REPO="${HOME}/cs8803-grpo-ablation"
SCRATCH_ROOT="$(readlink -f "${HOME}/scratch")/grpo"
export HF_HOME="${SCRATCH_ROOT}/hf"
export HF_DATASETS_CACHE="${SCRATCH_ROOT}/datasets"
export PYTHONPATH="${REPO}/src"
export TOKENIZERS_PARALLELISM=false
cd "${REPO}"
"${SCRATCH_ROOT}/envs/grpo-ice-train/bin/python" scripts/eval_ice_grpo.py \
  --config configs/ice_olmo2_7b_grpo.yaml --label step100 \
  --model "${SCRATCH_ROOT}/runs/olmo2-7b-grpo-100/model" \
  --split validation --output-dir "${SCRATCH_ROOT}/eval/olmo2-7b-80-10-10"
