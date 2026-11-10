"""Compare two GRPO clipping bounds on one ICE GPU using reused rollouts."""

import argparse
import json
from pathlib import Path

import torch
import trl
import yaml
from transformers import AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer

from gsm8k_split import load_split
from train_ice_ablation import SPLIT_SHA, write_json
from train_ice_grpo import build_splits, reward_func

ROOT = Path(__file__).resolve().parents[1]
BOUNDS = (0.20, 0.28)


class ClipProbeTrainer(GRPOTrainer):
    """Observe the exact forward pass used for loss without changing it."""

    def _compute_loss(self, model, inputs):
        if "old_per_token_logps" not in inputs:
            raise RuntimeError("Clip probe requires frozen rollout log probabilities")
        self._probe_inputs = inputs
        try:
            loss = super()._compute_loss(model, inputs)
        finally:
            self._probe_inputs = None
        record = self.clip_records[-1]
        expected = record["upper_clipped_120" if self.epsilon_high == BOUNDS[0] else "upper_clipped_128"]
        observed = self._metrics["train"]["clip_ratio/high_mean"][-1]
        if abs(observed - expected / record["valid_tokens"]) > 1e-6:
            raise RuntimeError("Clip probe disagrees with TRL's actual clipping metric")
        return loss

    def _get_per_token_logps_and_entropies(self, *args, **kwargs):
        logps, entropies = super()._get_per_token_logps_and_entropies(*args, **kwargs)
        inputs = getattr(self, "_probe_inputs", None)
        if inputs is not None:
            with torch.no_grad():
                ratios = (logps.detach() - inputs["old_per_token_logps"]).exp()
                valid = inputs["completion_mask"].bool()
                positive = valid & (inputs["advantages"].unsqueeze(1) > 0)
                count120 = (positive & (ratios > 1.20)).sum()
                count128 = (positive & (ratios > 1.28)).sum()
                tokens, n120, n128 = torch.stack((valid.sum(), count120, count128)).tolist()
            self.clip_records.append({
                "step": self.state.global_step + 1,
                "valid_tokens": tokens,
                "upper_clipped_120": n120,
                "upper_clipped_128": n128,
            })
        return logps, entropies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon-high", type=float, choices=BOUNDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=8)
    cli = parser.parse_args()
    if cli.iterations < 2 or cli.steps < cli.iterations or cli.steps % cli.iterations:
        parser.error("steps must be a positive multiple of iterations, and iterations must be at least two")
    if trl.__version__ != "0.28.0" or not torch.cuda.is_available():
        raise RuntimeError("Run on ICE with the study's TRL 0.28.0 GPU environment")

    cfg = yaml.safe_load((ROOT / "configs/ice_olmo_grpo.yaml").read_text())
    set_seed(cfg["seed"])
    source, split, manifest = load_split(cfg)
    if manifest["sha256"] != SPLIT_SHA:
        raise ValueError("The dataset split differs from the 1B study")
    train, _ = build_splits(source, split)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["model_revision"], local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    args = GRPOConfig(
        output_dir=str(cli.output),
        seed=cfg["seed"],
        max_steps=cli.steps,
        # One complete G=4 group per update. Multiple updates reuse its old logprobs.
        per_device_train_batch_size=cfg["group_size"],
        gradient_accumulation_steps=1,
        num_generations=cfg["group_size"],
        num_iterations=cli.iterations,
        max_completion_length=cfg["max_completion_length"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type="constant",
        beta=cfg["beta"],
        epsilon=cfg["epsilon"],
        epsilon_high=cli.epsilon_high,
        importance_sampling_level="token",
        loss_type="grpo",
        scale_rewards="group",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        model_init_kwargs={"revision": cfg["model_revision"], "dtype": torch.float32, "local_files_only": True},
    )
    trainer = ClipProbeTrainer(
        model=cfg["model"], reward_funcs=reward_func, args=args,
        train_dataset=train, processing_class=tokenizer,
    )
    trainer.clip_records = []
    result = trainer.train()
    if result.global_step != cli.steps or len(trainer.clip_records) != cli.steps:
        raise RuntimeError("Expected a clipping measurement for every optimizer update")
    first = next(
        (r for r in trainer.clip_records if r["upper_clipped_120"] > r["upper_clipped_128"]),
        None,
    )
    summary = {
        "gpu": torch.cuda.get_device_name(),
        "epsilon_low": cfg["epsilon"],
        "epsilon_high": cli.epsilon_high,
        "optimizer_steps": result.global_step,
        "iterations_per_rollout": cli.iterations,
        "first_clip_higher_witness": first,
        "records": trainer.clip_records,
    }
    cli.output.mkdir(parents=True, exist_ok=True)
    write_json(cli.output / "clip-probe.json", summary)
    print(json.dumps({key: val for key, val in summary.items() if key != "records"}), flush=True)


if __name__ == "__main__":
    main()
