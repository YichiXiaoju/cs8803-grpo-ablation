#!/bin/bash
#SBATCH --job-name=olmo-base-test
#SBATCH --account=ece
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail
REPO="${HOME}/cs8803-grpo-ablation"
SCRATCH_ROOT="$(readlink -f "${HOME}/scratch")/grpo"
export HF_HOME="${SCRATCH_ROOT}/hf"
export HF_DATASETS_CACHE="${SCRATCH_ROOT}/datasets"
export PYTHONPATH="${REPO}/src"
export TOKENIZERS_PARALLELISM=false
cd "${REPO}"
"${SCRATCH_ROOT}/envs/grpo-ice-train/bin/python" scripts/eval_ice_grpo.py \
  --label base --model allenai/OLMo-2-0425-1B --split test \
  --output-dir "${SCRATCH_ROOT}/eval/80-10-10"
