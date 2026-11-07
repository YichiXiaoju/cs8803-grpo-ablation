#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ "${SLURM_GPUS_ON_NODE:-}" != "2" ]]; then
  echo "This full-parameter FSDP run requires an allocation with 2 GPUs." >&2
  exit 1
fi

scratch_root="$(readlink -f "$HOME/scratch")/grpo"
export HF_HOME="$scratch_root/hf"
export HF_DATASETS_CACHE="$scratch_root/datasets"
export PYTHONPATH="$PWD/src"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1

# This watcher keeps only the step-100 model weights for later evaluation.
nohup "$scratch_root/envs/grpo-ice-train/bin/python" \
  scripts/preserve_7b_step100.py \
  >> "$scratch_root/runs/olmo2-7b-full-grpo-1epoch-lr1e-6-preserve-step100.log" 2>&1 &

exec "$scratch_root/envs/grpo-ice-train/bin/torchrun" \
  --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_ice_grpo.py \
  --config configs/ice_olmo2_7b_full_grpo.yaml \
  --output "$scratch_root/runs/olmo2-7b-full-grpo-1epoch-lr1e-6"
