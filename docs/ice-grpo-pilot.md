# ICE OLMo GRPO pilot

This runs a full-parameter, four-sample GRPO pilot on `allenai/OLMo-2-0425-1B` Base. It is an initial training check, not a final ablation result. Code stays in home; Conda environment, Hugging Face cache, and model outputs stay in scratch. The model-selection questions are excluded from this pilot's train and validation splits.

On the ICE login shell, after cloning the repository into `~/cs8803-grpo-ablation` and checking out the pilot branch:

```bash
module load anaconda3/2023.03
SCRATCH_ROOT="$(readlink -f ~/scratch)/grpo"
mkdir -p "$SCRATCH_ROOT/envs" "$SCRATCH_ROOT/conda-pkgs" "$SCRATCH_ROOT/pip-cache"
export CONDA_PKGS_DIRS="$SCRATCH_ROOT/conda-pkgs"
export PIP_CACHE_DIR="$SCRATCH_ROOT/pip-cache"
cd ~/cs8803-grpo-ablation
conda env create -p "$SCRATCH_ROOT/envs/grpo-ice-train" -f environment-ice-train.yml
sbatch --output="$SCRATCH_ROOT/olmo-grpo-%j.log" scripts/ice_olmo_grpo.sbatch
```

Check `squeue -u "$USER"`, then inspect the Slurm log and `~/scratch/grpo/runs/olmo-grpo-JOBID/summary.json`. The manifest records exact dataset rows and model revision. A successful pilot needs nonzero steps, finite loss, and changed model parameters. If every group is all-correct or all-incorrect, binary rewards provide no GRPO gradient; inspect reward and mixed-group logs before extending training.

Configuration explicitly selects `loss_type="grpo"` because recent TRL defaults to DAPO. Group standard-deviation scaling is on; beta is zero, so this pilot has no reference KL penalty. Group size is four and LoRA is unused. The resulting checkpoint is scratch-only and is ignored by Git.
