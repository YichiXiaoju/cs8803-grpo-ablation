# CS 8803 DRL — GRPO / Dr. GRPO / DAPO Ablations

**Status: reproducible objective-layer starter, NOT an end-to-end LLM trainer.**

Course team project comparing (1) baseline Group Relative Policy Optimization (GRPO), (2) Dr. GRPO normalization changes, and (3) DAPO components: clip-higher, dynamic sampling, token-level loss and overlong response handling.

## Research questions

1. What do response-wise vs. token-wise vs. fixed-length loss aggregation change?
2. How does standardizing group rewards affect difficulty-dependent gradient weighting?
3. What is the independent and combined effect of clip-higher and dynamic sampling under matched rollout budgets?

## Proposed experiment
- **Initial policy:** to be selected after a fixed-prompt comparison of ICE-compatible base checkpoints.
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

Model and dataset downloads go to `~/.cache/huggingface` by default, outside this repository. The script rejects `HF_HOME` or `HF_HUB_CACHE` pointing into the repository. Local `runs/`, `models/`, checkpoint, and weight files are Git ignored. Keep any manually downloaded model under `models/` or outside the repository.

For a fresh environment after changing dependencies:

```bash
conda env update -f environment.yml --prune
```

## Repository map

```text
src/grpo_ablation/objective.py  # Advantage, clipping and three loss denominators
src/grpo_ablation/rewards.py    # Strict GSM8K final-number verifier
src/grpo_ablation/configs.py    # Component presets
scripts/inspect_gsm8k.py       # Dataset inspection utility
scripts/model_selection.py     # Resumable five-model comparison
scripts/summarize_model_selection.py  # Validate and summarize results
studies/model_selection/       # Fixed protocol, question IDs, compact results
environment.yml                # Conda environment for Apple Silicon evaluation
tests/                         # Mathematical and gradient-flow unit tests
docs/EXPERIMENT_PLAN.md         # Controlled design and metrics
docs/TEAM.md                    # Four-person ownership
```

## Important limitations / next milestones

- The model-selection evaluation has policy generation but no log-prob recomputation, optimizer training loop or GRPO weight updates yet.
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
