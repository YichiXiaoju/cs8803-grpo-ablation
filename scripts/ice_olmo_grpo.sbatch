#!/bin/bash
#SBATCH --job-name=olmo-grpo-pilot
#SBATCH --account=ece
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00

set -euo pipefail
REPO="${HOME}/cs8803-grpo-ablation"
SCRATCH_ROOT="$(readlink -f "${HOME}/scratch")/grpo"
export HF_HOME="${SCRATCH_ROOT}/hf"
export HF_DATASETS_CACHE="${SCRATCH_ROOT}/datasets"
export PIP_CACHE_DIR="${SCRATCH_ROOT}/pip-cache"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${SCRATCH_ROOT}/runs"
cd "${REPO}"
"${SCRATCH_ROOT}/envs/grpo-ice-train/bin/python" scripts/train_ice_grpo.py \
  --config configs/ice_olmo_grpo.yaml \
  --output "${SCRATCH_ROOT}/runs/olmo-grpo-80-10-10"
