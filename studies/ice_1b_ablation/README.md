# OLMo 2 1B normalization and sampling ablations

These are the single-seed ICE runs based on PR #5 (`d958c98`). All arms start from the same OLMo 2 1B base revision, use full-parameter training on one H100, and share the pinned GSM8K 80/10/10 split, training seed 42, four completions per prompt, a 192-token cap, and binary exact-answer rewards. The split is a reshuffle of the original GSM8K train and test rows. The figures below use our held-out validation split.

| Arm | Treatment relative to E0 | ICE training job | Validation job |
| --- | --- | ---: | ---: |
| E0 | GRPO core, group reward standardization and per-response loss mean | archived 7,034-step run | archived validation |
| E1 | Fixed 192-token loss denominator | 6906917 | 6914099 |
| E2 | Center rewards without dividing by group standard deviation | 6906918 | 6914100 |
| E3 | E1 and E2 together | 6906919 | 6914106 |
| E5 | Resample groups when all four rewards agree | 6906920 | 6914107 |

[comparison.csv](comparison.csv) is the original aggregate output of ICE job 6914109. All rows use the same 879 validation questions in the same order, with four samples per question (seed 4242, temperature 0.8, top-p 0.95, top-k 0, BF16). Accuracy is the fraction of correct answers across all 3,516 samples. Mean answer length retokenizes decoded answer text with the pinned base tokenizer and excludes padding and special tokens.

The budget is 3,453,024 completion tokens along the E0 trajectory that produced the evaluated checkpoint. An abandoned E0 restart generated another 2,792 tokens. Each new arm stops after the group that reaches the target, so it can overshoot by up to one group. E5 counts rejected groups toward its budget. The trainer maps the E0 linear decay to cumulative token boundaries using `configs/e0_completion_tokens.csv`.

## Reproduce on ICE

From a checkout in your ICE account, submit the training jobs with the script below. It uses the `cs` allocation and logs its job IDs in `~/scratch/grpo-e0-ablation/jobs.tsv`. The launcher is idempotent and reuses existing study outputs and job IDs at that path.

```bash
bash scripts/submit_ice_olmo_1b_ablations.sh
```

Once training summaries show `budget_reached`, submit evaluation from the checkout. Set `E0_VALIDATION` to the archived `full-epoch-7034-validation.json` from the team result collection.

```bash
study="$(readlink -f ~/scratch)/grpo-e0-ablation"
for arm in E1 E2 E3 E5; do
  sbatch --job-name="gsm8k-$arm" --output="$study/logs/val-$arm-%j.log" \
    scripts/eval_ice_1b_ablation.sbatch "$arm" "$PWD"
done

# Run after all four validations complete, in an ICE CPU allocation.
HF_HOME="$study/hf" "$study/env/bin/python" scripts/summarize_ice_1b_ablation.py \
  --e0 "${E0_VALIDATION:?Set the archived validation JSON path}" \
  --study "$study" --output "$study/validation/comparison.csv"
```

ICE setup job 6906915 recorded Torch 2.15.0+cu132, Transformers 4.57.6 and TRL 0.28.0. Checkpoints and per-answer records are stored in scratch. The Git repository contains the audited token curve and aggregate metrics.
