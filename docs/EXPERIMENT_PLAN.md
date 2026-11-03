# CS 8803 DRL — Controlled Experiment Plan (Team T26)

**Status:** revised proposal for team review, 2026-11-02. No model-training results have been collected yet. Includes both **controlled component ablations** and a **DAPO-style full-recipe comparison**; none is an exact reproduction of the original papers.

**Research question:** How do group reward standardization, response-length aggregation, asymmetric clipping, and dynamic sampling influence mathematical reasoning accuracy, output length, exploration, and compute efficiency?

## 1. Scope and milestones

| Date | Required deliverable | Experiment expectation |
| --- | --- | --- |
| Nov 5 | Integratable GRPO baseline + Dr. GRPO normalization switches + DAPO interfaces | Objective unit tests, manual numerical checks; minimal end-to-end generation/reward/one update if GPU ready |
| Nov 11 (internal) / Nov 12 (Canvas) | Slides **PDF** and presentation **script** | Describe design and report only verified pilot findings; **no video required** |
| Nov 18 (internal) / Nov 19 (Canvas) | 12–15-minute revised video, Media Gallery link, 5 quiz questions | Include confirmed preliminary results where available |
| Dec 7 | Final video, implementation, modification, findings | Controlled training ablations and reproducibility artifacts |

**Priorities:** P0 = pilot and working baseline; P1 = E0–E3 normalization factorial **plus E7 token-level-only and E8 DAPO-style combined recipe**; P2 = E4/E5 isolated clipping/sampling comparisons; P3 = optional E6 clipping × sampling interaction. **E0, E3 and E8 are the three method-level reference arms.** Planned arms are subject to GPU/rollout feasibility; no results are assumed.

## 2. Common experimental setup (provisional)

| Dimension | Setting |
| --- | --- |
| Initial checkpoint | `Qwen/Qwen2.5-3B` **Base** (subject to pilot; use identical checkpoint for all arms) |
| Hardware | One full NVIDIA H200 if allocation permits (MIG partition and GPU-hours TBD) |
| Training | BF16 + LoRA, same LoRA targets/rank, optimizer, schedule and training framework |
| Dataset | `openai/gsm8k`, `main`; train-only pool for training and validation split; untouched official test for final evaluation |
| Reward | Rule-based final numerical correctness: correct=1, incorrect=0; no learned reward model |
| Group size | `G=8`, same prompts/decoding temperature (initially 0.8) and response format |
| Max completion length | Initially 512; pilot measures truncation and may trigger **one shared** adjustment |
| Baseline clipping | `epsilon_low=epsilon_high=0.20`; DAPO Clip-Higher uses `epsilon_high=0.28` |
| Reference/KL | Keep identical across arms; for first controlled study use no explicit reference KL (`beta=0`) and document this simplification |
| Training budget | Equal **attempted** prompt/rollout token budget as primary comparison; log equal-step comparisons separately; report generated tokens and GPU-hours |
| Evaluation | Identical held-out prompts and fixed decoding settings; measure accuracy and lengths at matched token-budget checkpoints |
| Seeds | One feasibility run; target >=3 seeds for P1 if GPU allocation allows; report uncertainty |

**Pilot gate (before locking arms):** 100 seeded GSM8K training prompts × 8 independent completions. Log final-answer parse success, per-response accuracy, `K=0` / `0<K<G` / `K=G` rates, response lengths by correctness, truncation rate and end-to-end tokens/sec. The objective is a nondegenerate mixed-reward rate, not a predefined threshold. If Base model rarely yields valid mixed groups, consider a **common** SFT warm start from train-only examples, or revise prompt/task once and then freeze conditions for *all* arms. No official test examples for training, model selection, or SFT.

## 3. P1: normalization factorial — E0–E3 (Xiaoju)

Two independent knobs: (a) divide by per-group reward standard deviation vs. center only; (b) divide each response by its own length `T_i` vs. fixed `L`. Other factors stay fixed; **do not enable Dynamic Sampling or Clip-Higher in E0–E3**.

| ID | Name | Advantage | Loss aggregation | Scientific question |
| --- | --- | --- | --- | --- |
| **E0** | Original GRPO core | `(r-mean)/std` | response mean: `(1/B) sum_i (1/T_i) sum_t ell_it` | Baseline |
| **E1** | Length fix only | `(r-mean)/std` | fixed: `(1/(B L)) sum_it ell_it` | Does removing response-specific `1/T_i` affect output length and accuracy? |
| **E2** | Std fix only | `r-mean` | response mean | Does removing `1/std` alter difficulty-dependent gradient weighting? |
| **E3** | Dr. GRPO core | `r-mean` | fixed | Joint effect and interaction of both changes |

Here `B=P*G`, `ell_it` is the same PPO-clipped **surrogate** for each experiment, and all equations are before the sign flip for optimizer loss. In real code use `std.clamp_min(eps)` and verify all-equal reward groups yield zero advantages. **Fixed length does not mean no normalization:** `L` is the configured maximum completion length.

**Comparisons:** E1−E0 = length effect with std on; E3−E2 = length effect with std off; E2−E0 = std effect with response mean; E3−E1 = std effect with fixed length. A difference-in-differences, `(E3-E2)-(E1-E0)`, tests interaction on a chosen outcome (accuracy, length, etc.), with caution about seed noise.

**Primary metrics:** held-out answer accuracy; mean and distribution of correct/incorrect response length; truncation; gradient norm, parameter-update norm; reward std and `K` frequency. Predefine difficulty buckets on **initial checkpoint** pilot pass rate and freeze those bucket assignments to avoid moving-label confounds.

## 4. DAPO component ablations — E4/E5 mandatory targets, E6 optional interaction (Kira)

Use the **same E0** as the fourth corner; keep reward standardization ON, response-mean aggregation and all other choices identical. This intentionally isolates clipping and sampling, rather than claiming full DAPO.

| ID | Name | Clip-Higher | Dynamic Sampling | Scientific question |
| --- | --- | --- | --- | --- |
| **E0** | Shared baseline | OFF (upper 1.20) | OFF | Baseline |
| **E4** | Clip-Higher only | ON (upper 1.28) | OFF | Does wider positive-advantage clipping change entropy / positive clip fraction / accuracy? |
| **E5** | Dynamic Sampling only | OFF | ON | Does filtering all-correct/all-wrong groups improve learning per *generated token*? |
| **E6 (optional)** | Both | ON | ON | Do the two effects interact? |

**Dynamic Sampling policy:** for binary rewards, accept a prompt group only when `0<K<G`; rejected groups consume generation compute and must be counted. State how the next prompt is drawn; make attempts per run finite and log acceptance/rejection rates, attempted prompts, generated tokens and wall-clock cost. Dynamic sampling changes the distribution of *accepted* prompts. Do **not** compare E5/E6 with E0 solely by equal optimizer steps; provide accuracy at equal attempted generation budgets.

**Secondary metrics:** policy/token entropy computed consistently, positive/negative clipped-token fractions, mixed-group rate before and after filtering, accepted difficulty distribution, and all attempted (including discarded) completion tokens.

## 5. Required DAPO token-level loss and combined comparison — E7/E8

DAPO's **token-level loss is a core method component**. It is not optional when claiming to compare with a DAPO-style recipe. The existing E4–E6 runs intentionally hold response-wise aggregation fixed to isolate clip/sampling effects; none should be labeled full DAPO.

| ID | Name | Relative to E0 | Purpose | Priority |
| --- | --- | --- | --- | --- |
| **E7** | Token-level mean only | Change **only** response mean to `sum_it ell_it / sum_i T_i` | Isolate effect of DAPO loss aggregation on length, accuracy, gradient scale | **Required target** |
| **E8** | DAPO-style full recipe | Clip-Higher, Dynamic Sampling, token-level mean, overlong filtering + soft overlong penalty | Compare combined DAPO-style method against E0 GRPO and E3 Dr. GRPO | **Required target** |

**Algorithm-level comparisons:** E0 vs E3 vs E8 answer which *combined configurations* perform differently in the same 3B/GSM8K setting. Differences between E8 and E0 **cannot be attributed to any one component**. Use E1/E2/E4/E5/E7 to isolate individual mechanisms; E6 is the optional clip×sampling interaction.

**Implement E8 carefully:** define a shared maximum length and an explicit overlong threshold; document which completions are excluded from policy loss, which still count toward sampling cost, and the soft penalty applied near the length cap. Treat overlong handling as distinct from normal answer correctness; avoid treating truncated generations as reliable negative labels. Use identical reward scale and evaluation settings across all arms.

**Offline fixed-rollout diagnostic (still required as a unit-test / analysis deliverable):** on identical `new_logp`, `old_logp`, rewards and token masks, compare `response` / `token` / `fixed` aggregation; verify scalar loss, gradient direction, per-token weights and batch-dependent scale. This is **not** a substitute for the E7 training comparison when resources permit.

For identical fixed rollouts/advantages, token mean versus fixed-`L` gradient differs by a batch-dependent positive scalar `B*L / sum_i T_i`; changing reward normalization breaks this equivalence.

**Feasibility fallback:** if E8 or E7 cannot be trained robustly within the available allocation, report the implementation/unit tests and exactly which arms were actually run, without claiming a completed full-method comparison.

## 6. Evaluation protocol and fairness

1. **Lock setup after pilot:** checkpoint/hash, GSM8K split indices, answer parser, prompt template, decoding temperature, `G`, max length, LoRA config, optimizer, LR schedule, loss config, seeds and hardware. Save configurations alongside logs.
2. **Unambiguous reward verifier:** extract the model's *final* answer, normalize numerical forms, audit random successes/failures manually; explicitly track parse failures and truncated completions. Do not reward a correct intermediate number if final answer is wrong.
3. **Separate three clocks:** optimizer steps, *attempted generated tokens*, and wall-clock/GPU-hours. For Dynamic Sampling, discarded rollouts count toward generation budget. Report both matched-token and matched-step graphs, labeling which is causal-comparable.
4. **Gradient scale caveat:** removing std and changing denominator may alter effective update magnitude even with identical LR. Log gradient / update norms and clipping frequencies; if practical report a scale-controlled sensitivity check rather than claiming a normalization effect independent of optimization scale.
5. **Evaluation settings:** fixed held-out validation and decoding settings (greedy and/or a declared fixed sampling protocol), same answer parser and max output length, checkpoint regularly. Official GSM8K test evaluated only after model selection.
6. **Uncertainty:** multiple seeds if feasible; report mean, dispersion and actual raw run values. Avoid overstating gains from a short pilot or one seed.
7. **Train/serve stack:** `src/grpo_ablation/objective.py` already implements objective switches and their tests; it is **not** an end-to-end trainer. FX/A establish generation, reward and LoRA optimizer; B/C integrate their mechanisms into the *same* trainer. Verify detached old policy / frozen advantages and valid-token masking.

## 7. Who does what

| Owner | By Nov 5 | By Nov 11/12 | Before final |
| --- | --- | --- | --- |
| **GRPO lead** | Baseline end-to-end interfaces, group generation/reward/one optimizer step | GRPO pilot, slides and script | Reproducible E0, baseline discussion |
| **Xiaoju — Dr. GRPO** | Verify `advantage=std/center`, `aggregation=response/fixed`; numerical and gradient tests | E1–E3 switches integrated, short validation, Dr. GRPO slides/script | E0–E3 length/difficulty analysis and causal comparisons |
| **Kira — DAPO** | Clip-Higher, Dynamic Sampling **and token-level loss** interfaces, unit tests | E7 single-change loss test, E8 overlong-handling specification, DAPO slides/script | E4/E5/E7 component results and E8 combined DAPO-style result with FX; E6 if resources permit |
| **FX — integration** | Decide H200 access, common checkpoint/splits/logging and GPU budget | Assemble full slides/script, pilot results labeled provisional | Integrate E0/E3/E8 comparisons, ensure token budget and overlong accounting, manage runs, report/video |

Video editing should be explicitly assigned separately; it is not automatically FX's responsibility.

## 8. Minimum results / stop-go rules

**Nov 12 Initial Draft:** submit PDF slides + script, with an accurate experimental plan and genuinely verified progress. **Do not require a completed video or fabricate ablation outcomes.**

**Dec 7 Final:** target E0/E3/E8 method comparison **and** E0–E3 normalization factorial, E4/E5/E7 component controls; E6 is optional if time/compute permits. This is an ambitious target, not a guarantee: always present the exact set actually run. If mixed-reward rate is near zero or rollout costs exceed budget, report feasibility findings, downscope model/prompt budget *for all arms*, or use fixed-rollout diagnostics and clearly label uncompleted work.

**Unit-test checklist:** numerical std vs centered advantages (including homogeneous groups), per-token clipping gradients at upper/lower boundaries, length-aggregation coefficients (including E7 token-level mean), fixed-`L` gradient scaling, masked padding zero gradient, frozen old logprobs/advantages, dynamic-sampling acceptance for K=0/1/G, overlong filtering/soft penalty and reward final-answer extraction.

## 9. References

- DeepSeekMath (GRPO): https://arxiv.org/abs/2402.03300
- Dr. GRPO: https://arxiv.org/abs/2503.20783
- DAPO: https://arxiv.org/abs/2503.14476
- TRL GRPOTrainer: https://huggingface.co/docs/trl/grpo_trainer

Team T26 dates/scope: [CS 8803 assignment spreadsheet](https://docs.google.com/spreadsheets/d/1aLjzTaep6zFwFXzyeeD0YCzK9Ttj8O5ndPJQ-6uC1q0/edit).
