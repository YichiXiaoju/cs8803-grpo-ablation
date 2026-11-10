"""Run a single-GPU PR #5 ablation at the 1B E0 completion-token budget."""

import argparse
import csv
import json
import random
import time
from bisect import bisect_right
from pathlib import Path

import torch
import trl
import yaml
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from grpo_ablation.overlong import DEFAULT_CACHE, OverlongShapedReward, completion_token_counts
from grpo_ablation.treatments import validate_treatment
from train_ice_grpo import build_splits, latest_complete_checkpoint, reward_func
from gsm8k_split import load_split, write_manifest

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 3_453_024
SPLIT_SHA = "709281297b46dab371e027be973bf7de17d17fbccfbc33c3e5a07607cff31d94"


class BudgetReached(Exception):
    """The final attempted group used the budget without an optimizer update."""


class AttemptLedger:
    def __init__(self, path, target):
        self.path = path
        self.target = target
        self.tokens = 0
        self.rejected_tokens = 0
        self.attempts = 0
        if path.exists():
            with path.open() as handle:
                for line in handle:
                    record = json.loads(line)
                    self._count(record["completion_tokens"], record["accepted"])

    def _count(self, tokens, accepted):
        self.tokens += tokens
        self.attempts += 1
        if not accepted:
            self.rejected_tokens += tokens

    def record(self, step, tokens, accepted):
        if not 4 <= tokens <= 768:
            raise ValueError(f"Invalid token count for a four-answer group: {tokens}")
        self._count(tokens, accepted)
        entry = {"optimizer_step_before": step, "completion_tokens": tokens,
                 "accepted": accepted, "cumulative_completion_tokens": self.tokens}
        with self.path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")


class StopOnBudget(TrainerCallback):
    def __init__(self, ledger, deadline):
        self.ledger = ledger
        self.deadline = deadline

    def on_step_end(self, args, state, control, **kwargs):
        if self.ledger.tokens >= self.ledger.target or time.monotonic() >= self.deadline:
            control.should_save = True
            control.should_training_stop = True
        return control


class AblationTrainer(GRPOTrainer):
    def __init__(self, *args, ledger, dynamic_sampling, e0_tokens, shaped=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.accelerator.num_processes != 1 or self.args.generation_batch_size != 4:
            raise ValueError("The E0 rollout protocol requires one process and four completions per generation.")
        self.ledger = ledger
        self.dynamic_sampling = dynamic_sampling
        self.e0_tokens = e0_tokens
        self.shaped = shaped  # OverlongShapedReward, used to judge groups by correctness

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            # E0 used a linear 7034-step decay. Its logged token boundaries give
            # the equivalent LR at the same generation budget for each variant.
            def factor(_step):
                e0_step = bisect_right(self.e0_tokens, self.ledger.tokens) - 1
                return max(0.0, 1.0 - e0_step / (len(self.e0_tokens) - 1))

            self.lr_scheduler = LambdaLR(optimizer or self.optimizer, factor)
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _generate_and_score_completions(self, inputs):
        if not self.model.training:
            return super()._generate_and_score_completions(inputs)
        if self.ledger.tokens >= self.ledger.target:
            raise BudgetReached
        step = self.state.global_step
        if len(inputs) != 4 or len({row["prompt"] for row in inputs}) != 1:
            raise ValueError("A rollout must contain four completions for exactly one prompt.")
        # Replacement prompts come from the same frozen training split. The
        # attempt number is durable across a Slurm continuation.
        rng = random.Random((self.args.seed << 32) + self.ledger.attempts)
        while True:
            result = super()._generate_and_score_completions(inputs)
            # Count from completion_ids: completion_mask is zeroed for truncated answers when
            # mask_truncated_completions=True, which would undercount attempted tokens.
            tokens = int(completion_token_counts(result["completion_ids"], self.eos_token_id).sum().item())
            if self.shaped is not None:
                # Shaped rewards make advantages nonzero even for all-correct groups; use correctness.
                correct = sum(self.shaped.last_correct)
                informative = 0 < correct < len(self.shaped.last_correct)
            else:
                informative = bool(result["advantages"].ne(0).any().item())
            accepted = not self.dynamic_sampling or informative
            self.ledger.record(step, tokens, accepted)
            if accepted:
                return result
            if self.ledger.tokens >= self.ledger.target:
                raise BudgetReached
            inputs = [self.train_dataset[rng.randrange(len(self.train_dataset))]] * 4


def token_curve(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    steps = [int(row["optimizer_step"]) for row in rows]
    counts = [int(row["cumulative_completion_tokens"]) for row in rows]
    if steps != list(range(7035)) or counts[0] != 0 or counts[-1] != BUDGET:
        raise ValueError("The token curve must match the audited E0 run.")
    if any(not 4 <= right - left <= 768 for left, right in zip(counts, counts[1:])):
        raise ValueError("The token curve contains invalid completion counts.")
    return counts


def write_json(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment-seconds", type=int, default=52_200)
    parser.add_argument("--smoke", action="store_true", help="Use a small rollout budget for integration validation.")
    parser.add_argument("--seed", type=int, help="Override the training seed (default: the baseline config seed).")
    cli = parser.parse_args()
    if trl.__version__ != "0.28.0":
        raise RuntimeError("This rollout hook requires the PR #5 pin, TRL 0.28.0.")
    cfg = yaml.safe_load((ROOT / "configs/ice_olmo_grpo.yaml").read_text())
    overrides = yaml.safe_load(cli.config.read_text())
    name = overrides["experiment"]
    treatment = validate_treatment(overrides)
    loss, scale, dynamic = overrides["loss_type"], overrides["scale_rewards"], overrides["dynamic_sampling"]
    if cli.seed is not None:
        cfg["seed"] = cli.seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Run inside a BF16-capable ICE GPU allocation.")
    target = 8192 if cli.smoke else BUDGET
    counts = token_curve(ROOT / "configs/e0_completion_tokens.csv")
    deadline = time.monotonic() + cli.segment_seconds
    output = cli.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source, split, split_manifest = load_split(cfg)
    if split_manifest["sha256"] != SPLIT_SHA:
        raise ValueError("The dataset split differs from the audited 1B E0.")
    write_manifest(output / "split_manifest.json", split_manifest)
    train, evaluation = build_splits(source, split)
    manifest = {
        "pr_head": "d958c98ae12383199912ae72f152a90a83c60ecd",
        "baseline_config": {**cfg, "max_steps": 7034},
        "treatment": overrides, "split_sha256": SPLIT_SHA,
        "target_completion_tokens": target, "learning_rate_schedule": "E0 linear schedule mapped to audited token boundaries",
    }
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Output already belongs to a different configuration.")
    write_json(path, manifest)

    trainer_dir = output / "trainer"
    checkpoint = latest_complete_checkpoint(trainer_dir)
    ledger = AttemptLedger(output / "rollout_tokens.jsonl", target)
    if ledger.attempts and not checkpoint:
        raise RuntimeError("Previous attempts exist without a complete checkpoint. Use a new output directory.")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["model_revision"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    args = GRPOConfig(
        output_dir=str(trainer_dir), run_name=f"olmo-1b-{name}",
        seed=cfg["seed"], max_steps=target // 4 + 1,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_generations=cfg["group_size"], max_completion_length=cfg["max_completion_length"],
        temperature=cfg["temperature"], top_p=cfg["top_p"], learning_rate=cfg["learning_rate"],
        beta=cfg["beta"], epsilon=cfg["epsilon"], epsilon_high=treatment["epsilon_high"],
        num_iterations=treatment["num_iterations"], loss_type=loss, scale_rewards=scale,
        mask_truncated_completions=treatment["overlong_filtering"], bf16=True, gradient_checkpointing=True,
        logging_steps=1, save_strategy="steps", save_steps=100, save_total_limit=2,
        report_to="none", remove_unused_columns=False,
        model_init_kwargs={"revision": cfg["model_revision"], "dtype": torch.float32},
    )
    shaped = None
    if treatment["overlong_penalty"]:
        shaped = OverlongShapedReward(reward_func, cfg["max_completion_length"], DEFAULT_CACHE)
    trainer = AblationTrainer(
        model=cfg["model"], reward_funcs=shaped or reward_func, args=args, shaped=shaped,
        train_dataset=train, eval_dataset=evaluation, processing_class=tokenizer,
        callbacks=[StopOnBudget(ledger, deadline)], ledger=ledger,
        dynamic_sampling=dynamic, e0_tokens=counts,
    )
    try:
        trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    except BudgetReached:
        # No group crossed into an optimizer update when the last attempt was rejected.
        pass
    finished = ledger.tokens >= target
    if finished:
        trainer.save_model(str(output / "model"))
    summary = {
        "experiment": name, "status": "budget_reached" if finished else "continue",
        "optimizer_steps": trainer.state.global_step,
        "target_completion_tokens": target, "attempted_completion_tokens": ledger.tokens,
        "rejected_completion_tokens": ledger.rejected_tokens, "attempted_groups": ledger.attempts,
        "overshoot_completion_tokens": max(0, ledger.tokens - target),
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
