# Four-person ownership

| Member | Role | Tasks | Deliverable |
|---|---|---|---|
| Jinlin | A / GRPO baseline | GRPO formulation, rollout/logprob interface, baseline evaluation | baseline training run, explanation, and tests |
| Xin (Kira) | B / DAPO | clip-higher, dynamic sampling, token-level loss, overlong handling | DAPO component comparisons and explanation |
| Yichi (Xiaoju) | C / Dr. GRPO | reward std and length normalization ablations | normalization comparisons and length analysis |
| Ziming (FX) | D / Integration | shared trainer, compute logs, evaluation, and slide integration | reproducible run table and presentation |

Jinlin and Ziming own the shared GRPO baseline. Xin and Yichi implement and run their comparisons on that same training pipeline and initial checkpoint, rather than maintaining separate baselines. The video editor is assigned separately.

Each member reviews a different member's PR. Shared review of reward parsing and controlled variable table.
