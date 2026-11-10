#!/usr/bin/env bash
#SBATCH --account=cs
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00

set -euo pipefail
arm="${1:?Specify E0, E1, E2, E3, E4, E4C, E5, E6, E7 or E8}"
seed="${3:-}"  # optional seed of the run to evaluate
case "$arm" in E0|E1|E2|E3|E4|E4C|E5|E6|E7|E8) ;; *) exit 2 ;; esac
run="$arm${seed:+-s$seed}"
scratch="$(readlink -f "$HOME/scratch")"
root="${2:-$scratch/grpo-e0-code}"
study="$scratch/grpo-e0-ablation"
output="$study/validation"
python="$study/env/bin/python"
export HF_HOME="$study/hf"
export HF_DATASETS_CACHE="$study/datasets"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$output/$run"
exec 9>"$output/$run/.eval.lock"
flock -n 9
"$python" -c 'import json, sys; r=json.load(open(sys.argv[1])); assert r["status"] == "budget_reached", r' "$study/runs/$run/summary.json"
cd "$root"
"$python" scripts/eval_ice_grpo.py --label "$run" --config configs/ice_olmo_grpo.yaml \
  --model "$study/runs/$run/model" --split validation --output-dir "$output/$run"
