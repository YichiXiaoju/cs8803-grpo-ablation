#!/bin/bash
#SBATCH --job-name=grpo-e4-clip
#SBATCH --account=cs
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00

set -euo pipefail
root="${1:?Specify the checkout directory}"
study="$(readlink -f "$HOME/scratch")/grpo-e0-ablation"
output="$study/e4-probe/$SLURM_JOB_ID"
export HF_HOME="$study/hf"
export HF_DATASETS_CACHE="$study/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
python="$study/env/bin/python"
cd "$root"
for bound in 0.20 0.28; do
  "$python" scripts/probe_ice_clip_higher.py --epsilon-high "$bound" --output "$output/$bound"
done
"$python" - "$output" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
symmetric, higher = [json.loads((out / bound / "clip-probe.json").read_text())
                     for bound in ("0.20", "0.28")]
assert symmetric["first_clip_higher_witness"] or higher["first_clip_higher_witness"], (
    "Neither run encountered a positive-advantage token between the two clipping bounds"
)
print("Clip-Higher verified on actual training forwards:", out, flush=True)
PY
