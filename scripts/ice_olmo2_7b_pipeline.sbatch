#!/bin/bash
#SBATCH --job-name=olmo2-7b-grpo
#SBATCH --account=ece
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
bash "${HOME}/cs8803-grpo-ablation/scripts/ice_olmo2_7b_pipeline.sh"
