# DAPO arms on the matched-budget 1B protocol

Adds E4, E4C, E6, E7, E8 and multi-seed runs to the protocol of this study (same split, `G=4`,
192-token cap, E0 completion-token budget, E0-mapped LR schedule, rejected groups counted).
Code: `src/grpo_ablation/treatments.py` (arm definitions), `src/grpo_ablation/overlong.py`,
`scripts/train_ice_ablation.py` (`--seed` added), `scripts/submit_ice_dapo_ablations.sh`.

| Arm | Change relative to E0 | Question |
| --- | --- | --- |
| E7 | `loss_type=dapo` (token-level mean) | Effect of token-level aggregation alone |
| E4C | `num_iterations=K`, `epsilon_high=0.20` | Control: multi-update schedule, symmetric clip |
| E4 | `num_iterations=K`, `epsilon_high=0.28` | Effect of Clip-Higher (compare with **E4C**, not E0) |
| E6 | E4 + dynamic sampling | Clip-Higher x dynamic sampling (compare with E4C, E4, E5) |
| E8 | E7 + E4 + dynamic sampling + overlong filtering + soft overlong penalty | DAPO-style full recipe (not attributable to one part) |

## Why E4 needs `num_iterations > 1`

TRL sets `old_logps = logps.detach()` when `gradient_accumulation_steps % (steps_per_generation * num_iterations) == 0`.
With one update per rollout the importance ratio is exactly 1, so no token is ever clipped: E0 logged
`clip_ratio/high_mean = 0` on every step, and changing `epsilon_high` changes nothing. With `num_iterations=K`
the same rollout is reused K times, so later updates see ratios away from 1. This also changes the update
schedule, hence the E4C control. `K` is set in the configs (`num_iterations`); pick it with
`scripts/ice_clip_higher_probe.sbatch` (a witness between 1.20 and 1.28 must appear).

## Notes on E8

* `overlong_filtering` = TRL `mask_truncated_completions`: truncated answers get zero loss.
  TRL then zeroes `completion_mask`, so the ledger counts tokens from `completion_ids` instead
  (`completion_token_counts`); masked or rejected completions still count toward the budget.
* `overlong_penalty` adds a soft penalty to the 0/1 reward: 0 up to 152 tokens, linear to -1 at 192
  (DAPO Eq. 13; buffer 40 = 20% of the cap, as 4,096/20,480 in DAPO).
* Dynamic sampling judges a group by correctness (0 < correct < 4), not by shaped rewards.
* Base-model E0 runs often stop emitting EOS (E1 averages 192 tokens). If most answers hit the cap,
  filtering removes most of the training signal; check `completions/clipped_ratio` before reading E8.

## Run

```bash
bash scripts/submit_ice_dapo_ablations.sh primary   # smoke(E8), then E7 E4C E4 E6 E8
bash scripts/submit_ice_dapo_ablations.sh seeds     # seeds 43, 44 for E0, E5, E7 (runs/<arm>-s<seed>)
sbatch scripts/eval_ice_1b_ablation.sbatch E7 "$PWD"          # evaluate a run
sbatch scripts/eval_ice_1b_ablation.sbatch E0 "$PWD" 43       # evaluate a seed run
python scripts/summarize_ice_1b_ablation.py --arms E0 E1 E2 E3 E5 E7 E4C E4 E6 E8 \
  --e0 "$E0_VALIDATION" --study "$study" --output "$study/validation/comparison.csv"
```

All arms are single-seed unless a `-s<seed>` run exists; report seed spread before comparing arms.
The `E0-s43`/`E0-s44` runs use this trainer (E0 token budget), the archived E0 is the original run.

## Not yet validated on GPU

Unit tests cover the arm definitions, penalty, token counting and the rollout hook with TRL stubbed.
The first GPU run of each new path is `submit_ice_dapo_ablations.sh primary` (smoke on E8).
