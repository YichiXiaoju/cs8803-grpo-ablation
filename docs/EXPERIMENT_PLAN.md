# Experimental Plan (draft)

## Pilot before training
1. Select 100 training prompts using a fixed seed. Do not use GSM8K test set for model selection.
2. Generate 8 independent completions each with identical temperature, prompt template and token limit.
3. Measure baseline exact-match accuracy, output parsing rate, truncation rate, average correct/incorrect length and mixed-reward group rate.
4. If all-wrong groups dominate, change task difficulty, prompting or initial model **before** locking ablation conditions.

## Primary controlled 2x2 ablation
Hold checkpoint, prompt pool, reward, group size, clipping, max length, optimizer and training budget fixed.

| Run | Group std | Length aggregation |
|---|---|---|
| A Original GRPO core | On | Per-response mean |
| B Length fix only | On | Fixed L |
| C Std fix only | Off | Per-response mean |
| D Dr. GRPO core | Off | Fixed L |

The fixed-L objective and token-mean objective differ by batch-dependent scale. Optionally run a token-mean control to isolate DAPO aggregation. Do not claim the above presets fully reproduce any paper.

## DAPO add-on (if compute allows)
2x2 clipping (symmetric / clip-higher) x sampling (fixed prompt batches / dynamic resampling). Dynamic resampling must count discarded rollout tokens. Keep normalization and other factors fixed.

## Measurements
Validation exact-match, generated tokens, GPU-hours, update steps, KL, policy entropy, gradient/update norms, mean response length stratified by correctness, truncation, reward homogeneous group rate, accepted group difficulty distribution. At least two/three seeds when feasible.

## Pitfalls
- Positive-score reward groups with zero variance have zero centered advantage; dynamic sampling changes which prompts contribute updates.
- Length-based loss scale is not equivalent to removing group reward std.
- Compare at matched token budget and wall-clock in addition to optimizer steps.
- Answer parser needs audit: detect false positives, false negatives and malformed responses.
- Baseline uses an *Instruct* policy for tractability, unlike the Qwen2.5-32B Base starting point of DAPO.
