"""Small full-parameter GRPO pilot on ICE; writes all outputs to scratch."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from decimal import Decimal

import torch
import yaml
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from grpo_ablation.rewards import exact_number_reward, gsm8k_ground_truth
from gsm8k_split import load_split, write_manifest

ROOT = Path(__file__).resolve().parents[1]
PROMPT = (
    "Solve the math problem step by step. End with a separate line in the "
    "format Final answer: <number>.\n\n"
    "Question: Ana has 2 pens and buys 3 more. How many pens does she have?\n"
    "Solution: 2 + 3 = 5.\nFinal answer: 5\n\n"
    "Question: {question}\nSolution:"
)


def build_splits(source, split):
    def make(keys):
        data = []
        for part, idx in keys:
            row = source[part][idx]
            truth = gsm8k_ground_truth(row["answer"])
            if truth is None:
                raise ValueError(f"Unparseable ground truth at GSM8K {part} row {idx}")
            data.append({"prompt": PROMPT.format(question=row["question"]), "ground_truth": str(truth)})
        return Dataset.from_list(data)

    return make(split["train"]), make(split["validation"])


def reward_func(completions, ground_truth, **_kwargs):
    return [exact_number_reward(text, Decimal(truth)) for text, truth in zip(completions, ground_truth, strict=True)]


def latest_complete_checkpoint(trainer_dir):
    """Ignore a checkpoint interrupted while it was being written."""
    checkpoints = sorted(
        trainer_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[1]),
        reverse=True,
    )
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
    for path in checkpoints:
        if not all((path / name).is_file() for name in required):
            continue
        if (path / "model.safetensors").is_file():
            return path
        index = path / "model.safetensors.index.json"
        if index.is_file():
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if shards and all((path / shard).is_file() for shard in shards):
                return path
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ice_olmo_grpo.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, help="Extend an existing run to this many optimizer steps")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        cfg["max_steps"] = args.max_steps
    output = args.output.expanduser().resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Training outputs must be outside the Git repository")
    if not os.environ.get("HF_HOME") or Path(os.environ["HF_HOME"]).resolve().is_relative_to(ROOT):
        raise RuntimeError("Set HF_HOME to scratch outside the Git repository")
    if not torch.cuda.is_available():
        raise RuntimeError("Run inside a Slurm GPU allocation")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This pilot requires a GPU with bfloat16 support")
    if cfg["group_size"] != 4:
        raise ValueError("The agreed pilot uses G=4")

    output.mkdir(parents=True, exist_ok=True)
    source, split, split_manifest = load_split(cfg)
    write_manifest(output / "split_manifest.json", split_manifest)
    train, evaluation = build_splits(source, split)
    manifest = {
        "config": cfg, "split_sha256": split_manifest["sha256"],
        "train_questions": len(split["train"]), "validation_questions": len(split["validation"]),
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        previous_steps = previous["config"]["max_steps"]
        previous["config"]["max_steps"] = cfg["max_steps"]
        if previous != manifest or cfg["max_steps"] < previous_steps:
            raise ValueError("Existing run manifest differs; use a new output directory")
        if cfg["max_steps"] > previous_steps:
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["model_revision"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lora = cfg.get("lora")
    peft_config = None
    if lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=lora["r"], lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"], target_modules=lora["target_modules"],
            task_type="CAUSAL_LM",
        )
    args = GRPOConfig(
        output_dir=str(output / "trainer"),
        run_name=cfg["model"].split("/")[-1].lower() + "-grpo",
        seed=cfg["seed"],
        max_steps=cfg["max_steps"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=cfg["group_size"],
        max_completion_length=cfg["max_completion_length"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        learning_rate=cfg["learning_rate"],
        beta=cfg["beta"],
        epsilon=cfg["epsilon"],
        loss_type="grpo",
        scale_rewards="group",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=1 if cfg["max_steps"] <= 4 else 5,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        # Keep trainable weights in fp32 so 1e-6 updates are not rounded away.
        model_init_kwargs={"revision": cfg["model_revision"], "dtype": torch.bfloat16 if lora else torch.float32},
    )
    trainer = GRPOTrainer(
        model=cfg["model"], reward_funcs=reward_func, args=args,
        train_dataset=train, eval_dataset=evaluation, processing_class=tokenizer,
        peft_config=peft_config,
    )
    first_parameter = next(parameter for parameter in trainer.model.parameters() if parameter.requires_grad)
    before = hashlib.sha256(first_parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    checkpoint = latest_complete_checkpoint(output / "trainer")
    if checkpoint:
        print(f"Resuming from {checkpoint}", flush=True)
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    after = hashlib.sha256(first_parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    summary = {
        "training_performed": True, "steps": result.global_step,
        "train_loss": result.training_loss, "first_parameter_sha256_before": before,
        "first_parameter_sha256_after": after, "first_parameter_changed": before != after,
        "gpu": torch.cuda.get_device_name(0),
        "peak_cuda_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "resumed_from_checkpoint": str(checkpoint) if checkpoint else None,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    trainer.save_model(str(output / "model"))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
