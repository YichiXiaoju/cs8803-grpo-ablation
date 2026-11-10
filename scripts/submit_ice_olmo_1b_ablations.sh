#!/bin/bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="$(readlink -f "$HOME/scratch")/grpo-e0-ablation"
mkdir -p "$scratch/logs"
exec 8>"$scratch/.submit.lock"
flock -n 8
jobs="$scratch/jobs.tsv"
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
touch "$jobs"
setup="$(submit_once setup --output="$scratch/logs/setup-%j.log" "$root/scripts/ice_olmo_1b_ablation_setup.sbatch" "$root")"
train="$root/scripts/ice_olmo_1b_ablation.sbatch"
smoke="$(submit_once smoke --dependency="afterok:$setup" --time=00:30:00 \
  --job-name=grpo-1b-smoke --output="$scratch/logs/smoke-%j.log" "$train" E5 "$root" smoke)"
for arm in E1 E2 E3 E5; do
  submit_once "$arm" --dependency="afterok:$smoke" --job-name="grpo-1b-$arm" \
    --output="$scratch/logs/$arm-%j.log" "$train" "$arm" "$root" >/dev/null
done
cat "$jobs"
