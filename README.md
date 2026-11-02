# CS 8803 DRL — GRPO / Dr. GRPO / DAPO Ablations

**Status: reproducible objective-layer starter, NOT an end-to-end LLM trainer.**

Course team project comparing (1) baseline Group Relative Policy Optimization (GRPO), (2) Dr. GRPO normalization changes, and (3) DAPO components: clip-higher, dynamic sampling, token-level loss and overlong response handling.

## Research questions

1. What do response-wise vs. token-wise vs. fixed-length loss aggregation change?
2. How does standardizing group rewards affect difficulty-dependent gradient weighting?
3. What is the independent and combined effect of clip-higher and dynamic sampling under matched rollout budgets?

## Proposed experiment
- **Initial policy:** `Qwen/Qwen2.5-1.5B-Instruct` (pilot candidate; not yet benchmarked).
- **Task:** `openai/gsm8k` (`main`).
- **Reward:** exact final numeric answer correctness, binary 0/1; no reward model or critic training.
- **Pilot:** 100 train prompts × 8 responses, to verify mixed-group rate, parser accuracy and truncation before large training.
- **Main ablation:** `reward_std {on,off}` × `length_aggregation {per_response,fixed}`. Further DAPO-specific clip/sampling ablation if resources permit.

The Instruct checkpoint is a practical course choice, not a faithful reproduction of DAPO's 32B Base setup. Results should be described as **component ablations**, not official-paper replication.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e '.[dev]'
pytest -q
```

For GSM8K inspection:

```bash
pip install -e '.[data]'
python scripts/inspect_gsm8k.py
```

## Repository map

```text
src/grpo_ablation/objective.py  # Advantage, clipping and three loss denominators
src/grpo_ablation/rewards.py    # Strict GSM8K final-number verifier
src/grpo_ablation/configs.py    # Component presets
scripts/inspect_gsm8k.py       # Dataset inspection utility
configs/pilot.yaml             # Pilot configuration (not executed yet)
tests/                         # Mathematical and gradient-flow unit tests
docs/EXPERIMENT_PLAN.md         # Controlled design and metrics
docs/TEAM.md                    # Four-person ownership
```

## Important limitations / next milestones

- No policy generation, log-prob recomputation, LoRA, optimizer training loop or GPU benchmark yet.
- `dapo_core` toggles only token mean and asymmetric clipping: it is **not** full DAPO. Dynamic sampling and overlong handling remain to be implemented and evaluated.
- These objective functions omit optional KL and multi-GPU normalization details; use one consistently controlled training framework for end-to-end comparisons.
- Numeric reward verifier is intentionally strict; audit parsing failures. If you change model output format, update parser and tests together.
- Add CI, reproducible experiment scripts, metrics and checkpoints only after validating the pilot.

## References
- DeepSeekMath (GRPO): https://arxiv.org/abs/2402.03300
- Dr. GRPO: https://arxiv.org/abs/2503.20783
- DAPO: https://arxiv.org/abs/2503.14476
- TRL GRPOTrainer docs: https://huggingface.co/docs/trl/grpo_trainer

See [experiment plan](docs/EXPERIMENT_PLAN.md) and [team roles](docs/TEAM.md).
