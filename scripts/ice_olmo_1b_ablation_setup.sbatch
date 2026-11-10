#!/bin/bash
#SBATCH --job-name=grpo-1b-setup
#SBATCH --account=cs
#SBATCH --partition=coc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail
root="${1:?Specify the unpacked PR #5 directory}"
module load anaconda3/2023.03
scratch="$(readlink -f "$HOME/scratch")/grpo-e0-ablation"
export CONDA_PKGS_DIRS="$scratch/conda-pkgs"
export PIP_CACHE_DIR="$scratch/pip-cache"
export HF_HOME="$scratch/hf"
export HF_DATASETS_CACHE="$scratch/datasets"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"
cd "$root"
python="$scratch/env/bin/python"
if [[ ! -x "$python" ]]; then
  conda env create -p "$scratch/env" -f environment-ice-train.yml
fi
"$python" - <<'PY'
import torch, transformers, trl
from huggingface_hub import snapshot_download
from datasets import load_dataset
assert trl.__version__ == "0.28.0"
print(torch.__version__, transformers.__version__, trl.__version__, flush=True)
snapshot_download("allenai/OLMo-2-0425-1B", revision="a1847dff35000b4271fa70afc5db10fd29fedbdf")
load_dataset("openai/gsm8k", "main", revision="740312add88f781978c0658806c59bc2815b9866")
PY
