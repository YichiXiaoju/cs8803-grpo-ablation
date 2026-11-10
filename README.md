# CS 8803 DRL — GRPO / Dr. GRPO / DAPO Ablations

**Status: model-selection study complete; OLMo GRPO training pilot submitted on ICE, awaiting GPU validation.**

Course team project comparing (1) baseline Group Relative Policy Optimization (GRPO), (2) Dr. GRPO normalization changes, and (3) DAPO components: clip-higher, dynamic sampling, token-level loss and overlong response handling.

## Research questions

1. What do response-wise vs. token-wise vs. fixed-length loss aggregation change?
2. How does standardizing group rewards affect difficulty-dependent gradient weighting?
3. What is the independent and combined effect of clip-higher and dynamic sampling under matched rollout budgets?

## Proposed experiment
- **Initial policy:** `allenai/OLMo-2-0425-1B` Base, selected from the fixed-prompt comparison of ICE-compatible checkpoints.
- **Task:** `openai/gsm8k` (`main`).
- **Reward:** exact final numeric answer correctness, binary 0/1; no reward model or critic training.
- **Model selection study:** compare fixed GSM8K train prompts with four responses per prompt before choosing an ICE-compatible training checkpoint.
- **Main ablation:** `reward_std {on,off}` × `length_aggregation {per_response,fixed}`. Further DAPO-specific clip/sampling ablation if resources permit.

The eventual checkpoint will not reproduce DeepSeekMath's math pretraining or SFT. Its ability to follow the answer format and produce mixed-reward groups must be checked before training. Results should be described as **component ablations**, not official-paper replication.

## Initial model selection on Apple Silicon

```bash
conda env create -f environment.yml
conda activate grpo-pilot
pytest -q
python scripts/model_selection.py --model gemma
python scripts/model_selection.py --model smollm
python scripts/model_selection.py --model olmo
python scripts/model_selection.py --model qwen
python scripts/model_selection.py --model qwen_1_5b
python scripts/summarize_model_selection.py
```

The study uses the same 100 GSM8K training questions and four sampled responses per question for all five checkpoints. It reports exact-answer accuracy, mixed-reward group rate, parse rate, truncation and generation time. The fixed protocol and compact results are in [`studies/model_selection/`](studies/model_selection/README.md); raw responses remain in ignored `runs/model_selection/`. Both Qwen sizes use 4-bit MLX copies and are local-only because Qwen is prohibited on ICE. The other three models use bfloat16 on Apple MPS. This is inference only: no model weights are updated.

The ICE experiment combines the pinned GSM8K train and test source rows, then makes one deterministic 80/10/10 train/validation/test split (`scripts/gsm8k_split.py`, seed 42). The 100 questions previously used to select a model are included in **train**; model selection never used the validation or test questions. The split manifest, including all source indices and its SHA-256 hash, is written to the run/evaluation directory outside Git. `scripts/eval_ice_grpo.py` evaluates the full 10% validation split by default, with four generated answers each, temperature 0.8, top-p 0.95, top-k 0, and 192 new tokens. Compare base, step-4, and step-100 on that fixed validation split, reporting completion accuracy, first-sample accuracy, pass@4, mixed-reward group rate, parse rate, and truncation rate. Use `--questions 0 --split test` once for the final held-out evaluation. The earlier 128/20 pilot and its evaluation used a different split; its numbers cannot be compared directly with this experiment. Start this split in a new output directory (`scripts/ice_olmo_grpo.sbatch` uses `olmo-grpo-80-10-10`), then pass `--max-steps 100` with that same output to resume its 4-step checkpoint. The 4-step smoke test saves every step; longer runs save every 5 steps and keep the latest two complete checkpoints.

The OLMo 2 7B Base experiment uses the same split and generation settings. Its pinned model and LoRA settings are in [`configs/ice_olmo2_7b_grpo.yaml`](configs/ice_olmo2_7b_grpo.yaml). Run [`scripts/ice_olmo2_7b_pipeline.sh`](scripts/ice_olmo2_7b_pipeline.sh) inside an H100 allocation, or submit [`scripts/ice_olmo2_7b_pipeline.sbatch`](scripts/ice_olmo2_7b_pipeline.sbatch). The pipeline finishes the full 879-question Base validation evaluation before starting 100 GRPO optimizer steps. All weights, evaluation responses, and checkpoints stay in scratch; each evaluation question and training checkpoint can be resumed. Install `peft==0.17.1` in the ICE training environment before running the pipeline.

Model and dataset downloads go to `~/.cache/huggingface` by default, outside this repository. The script rejects `HF_HOME` or `HF_HUB_CACHE` pointing into the repository. Local `runs/`, `models/`, checkpoint, and weight files are Git ignored. Keep any manually downloaded model under `models/` or outside the repository.

For a fresh environment after changing dependencies:

```bash
conda env update -f environment.yml --prune
```

The [1B normalization and dynamic-sampling study](studies/ice_1b_ablation/README.md) records the matched-budget runs, validation results, and ICE commands.

## Repository map

```text
src/grpo_ablation/objective.py  # Advantage, clipping and three loss denominators
src/grpo_ablation/rewards.py    # Strict GSM8K final-number verifier
src/grpo_ablation/configs.py    # Component presets
scripts/inspect_gsm8k.py       # Dataset inspection utility
scripts/model_selection.py     # Resumable five-model comparison
scripts/summarize_model_selection.py  # Validate and summarize results
scripts/train_ice_grpo.py      # OLMo GRPO pilot training entry point
scripts/show_training_progress.py  # Report completed steps for a live run
scripts/ice_olmo_setup.sbatch  # ICE environment setup and model prefetch
scripts/ice_olmo_grpo.sbatch   # ICE GPU training job
studies/model_selection/       # Fixed protocol, question IDs, compact results
environment.yml                # Conda environment for Apple Silicon evaluation
environment-ice-train.yml      # Conda environment for ICE training
tests/                         # Mathematical and gradient-flow unit tests
docs/EXPERIMENT_PLAN.md         # Controlled design and metrics
docs/TEAM.md                    # Four-person ownership
```

## Important limitations / next milestones

- The model-selection evaluation is inference only. The separate ICE GRPO pilot has not yet produced a verified weight update or evaluation result.
- `dapo_core` toggles only token mean and asymmetric clipping: it is **not** full DAPO. Dynamic sampling and overlong handling remain to be implemented and evaluated.
- These objective functions omit optional KL and multi-GPU normalization details; use one consistently controlled training framework for end-to-end comparisons.
- Numeric reward verifier is intentionally strict; audit parsing failures. If you change model output format, update parser and tests together.
- Build and validate an end-to-end training pilot before comparing GRPO variants.

## References
- DeepSeekMath (GRPO): https://arxiv.org/abs/2402.03300
- Dr. GRPO: https://arxiv.org/abs/2503.20783
- DAPO: https://arxiv.org/abs/2503.14476
- TRL GRPOTrainer docs: https://huggingface.co/docs/trl/grpo_trainer

See [experiment plan](docs/EXPERIMENT_PLAN.md) and [team roles](docs/TEAM.md).

To check the latest complete checkpoint while a run is active:

```bash
python scripts/show_training_progress.py \
  --output /storage/ice1/1/5/jli3596/grpo/runs/olmo-1b-full-epoch-7034
```

Add `--watch 30` to poll every 30 seconds until the configured target is reached.
