#!/bin/bash
# Submit the DAPO arms on ICE, reusing the study layout and job log of submit_ice_olmo_1b_ablations.sh.
#   bash scripts/submit_ice_dapo_ablations.sh primary   # smoke(E8) -> E7, E4C, E4, E6, E8
#   bash scripts/submit_ice_dapo_ablations.sh seeds     # seeds 43 and 44 for E0, E5 and E7
set -euo pipefail
tier="${1:?Specify primary or seeds}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="$(readlink -f "$HOME/scratch")/grpo-e0-ablation"
mkdir -p "$scratch/logs"
exec 8>"$scratch/.submit.lock"
flock -n 8
jobs="$scratch/jobs.tsv"
touch "$jobs"
submit_once() {
  local name="$1" id
  shift
  id="$(awk -v key="$name" '$1 == key {print $2}' "$jobs")"
  if [[ -z "$id" ]]; then
    id="$(sbatch --parsable "$@")"
    id="${id%%;*}"
    printf '%s\t%s\n' "$name" "$id" >> "$jobs"
  fi
  printf '%s\n' "$id"
}
train="$root/scripts/ice_olmo_1b_ablation.sbatch"
case "$tier" in
  primary)
    # E8 exercises every new path (multi-update, Clip-Higher, dynamic sampling, overlong), so it is the smoke arm.
    smoke="$(submit_once smoke-E8 --time=00:30:00 --job-name=grpo-1b-smoke-E8 \
      --output="$scratch/logs/smoke-E8-%j.log" "$train" E8 "$root" smoke)"
    for arm in E7 E4C E4 E6 E8; do
      submit_once "$arm" --dependency="afterok:$smoke" --job-name="grpo-1b-$arm" \
        --output="$scratch/logs/$arm-%j.log" "$train" "$arm" "$root" >/dev/null
    done ;;
  seeds)
    for seed in 43 44; do
      for arm in E0 E5 E7; do
        submit_once "$arm-s$seed" --job-name="grpo-1b-$arm-s$seed" \
          --output="$scratch/logs/$arm-s$seed-%j.log" "$train" "$arm" "$root" train "$seed" >/dev/null
      done
    done ;;
  *) echo "Unknown tier: $tier" >&2; exit 2 ;;
esac
cat "$jobs"
