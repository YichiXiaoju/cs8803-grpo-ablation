#!/bin/bash
set -euo pipefail

REPO="${HOME}/cs8803-grpo-ablation"
SCRATCH_ROOT="$(readlink -f "${HOME}/scratch")/grpo"
PYTHON="${SCRATCH_ROOT}/envs/grpo-ice-train/bin/python"
export HF_HOME="${SCRATCH_ROOT}/hf"
export HF_DATASETS_CACHE="${SCRATCH_ROOT}/datasets"
export PYTHONPATH="${REPO}/src"
export TOKENIZERS_PARALLELISM=false
cd "${REPO}"

"${PYTHON}" scripts/eval_ice_grpo.py \
  --config configs/ice_olmo2_7b_grpo.yaml \
  --label base --model allenai/OLMo-2-1124-7B --split validation \
  --output-dir "${SCRATCH_ROOT}/eval/olmo2-7b-80-10-10"

OUTPUT="${SCRATCH_ROOT}/runs/olmo2-7b-grpo-100"
if [[ -f "${OUTPUT}/summary.json" ]] && \
   "${PYTHON}" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["steps"] < 100)' "${OUTPUT}/summary.json"; then
  echo "Training already completed 100 steps"
else
  "${PYTHON}" scripts/train_ice_grpo.py \
    --config configs/ice_olmo2_7b_grpo.yaml --output "${OUTPUT}" --max-steps 100
fi
