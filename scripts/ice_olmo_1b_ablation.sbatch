#!/bin/bash
#SBATCH --account=cs
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00

set -euo pipefail
arm="${1:?Specify E0, E1, E2, E3, E4, E4C, E5, E6, E7 or E8}"
root="${2:?Specify the unpacked PR #5 directory}"
mode="${3:-train}"
seed="${4:-}"  # optional training seed; output goes to runs/<arm>-s<seed>
case "$arm" in E0|E1|E2|E3|E4|E4C|E5|E6|E7|E8) ;; *) exit 2 ;; esac
case "$mode" in train|smoke) ;; *) exit 2 ;; esac
scratch="$(readlink -f "$HOME/scratch")/grpo-e0-ablation"
export HF_HOME="$scratch/hf"
export HF_DATASETS_CACHE="$scratch/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
python="$scratch/env/bin/python"
run="$arm${seed:+-s$seed}"
output="$scratch/runs/$run"
flags=()
if [[ -n "$seed" ]]; then flags+=(--seed "$seed"); fi
if [[ "$mode" == smoke ]]; then
  output="$scratch/runs/$run-smoke"
  flags+=(--smoke)
fi
mkdir -p "$output"
exec 9>"$output/.job.lock"
flock -n 9
if [[ -f "$output/summary.json" ]] && "$python" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["status"] != "budget_reached")' "$output/summary.json"; then
  exit 0
fi
cd "$root"
"$python" scripts/train_ice_ablation.py \
  --config "configs/ice_olmo_1b_$arm.yaml" --output "$output" "${flags[@]}"
status="$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$output/summary.json")"
if [[ "$mode" == smoke ]]; then
  "$python" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="budget_reached" and d["rejected_completion_tokens"]>0 and d["optimizer_steps"]>0, d' "$output/summary.json"
elif [[ "$status" == continue ]]; then
  sbatch --parsable --dependency="afterok:$SLURM_JOB_ID" --job-name="grpo-1b-$run" \
    --output="$scratch/logs/$run-%j.log" "$root/scripts/ice_olmo_1b_ablation.sbatch" "$arm" "$root" train "$seed"
fi
