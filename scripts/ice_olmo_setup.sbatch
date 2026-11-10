#!/bin/bash
#SBATCH --job-name=olmo-grpo-setup
#SBATCH --account=ece
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail
module load anaconda3/2023.03
SCRATCH_ROOT="$(readlink -f "${HOME}/scratch")/grpo"
export CONDA_PKGS_DIRS="${SCRATCH_ROOT}/conda-pkgs"
export PIP_CACHE_DIR="${SCRATCH_ROOT}/pip-cache"
export HF_HOME="${SCRATCH_ROOT}/hf"
mkdir -p "${SCRATCH_ROOT}/envs" "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}"
cd "${HOME}/cs8803-grpo-ablation"
if [[ ! -x "${SCRATCH_ROOT}/envs/grpo-ice-train/bin/python" ]]; then
  conda env create -p "${SCRATCH_ROOT}/envs/grpo-ice-train" -f environment-ice-train.yml
fi
"${SCRATCH_ROOT}/envs/grpo-ice-train/bin/python" -c 'import torch, trl, transformers, datasets; print("versions", torch.__version__, trl.__version__, transformers.__version__, datasets.__version__)'
"${SCRATCH_ROOT}/envs/grpo-ice-train/bin/python" - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="allenai/OLMo-2-0425-1B",
    revision="a1847dff35000b4271fa70afc5db10fd29fedbdf",
)
print(f"OLMo checkpoint cached at {path}", flush=True)
PY
