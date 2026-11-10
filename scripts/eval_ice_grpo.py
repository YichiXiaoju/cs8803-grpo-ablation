"""Resumable, paired GSM8K validation/test evaluation for GRPO checkpoints."""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_ablation.rewards import exact_number_reward, extract_final_number, gsm8k_ground_truth
from train_ice_grpo import PROMPT
from gsm8k_split import load_split, write_manifest

ROOT = Path(__file__).resolve().parents[1]
EVAL_SEED = 4242


def summarize(groups):
    completions = [item for group in groups for item in group["completions"]]
    n = len(completions)
    return {
        "questions": len(groups),
        "completions": n,
        "correct": sum(item["reward"] for item in completions),
        "completion_accuracy": sum(item["reward"] for item in completions) / n if n else None,
        "first_sample_accuracy": sum(group["completions"][0]["reward"] for group in groups) / len(groups) if groups else None,
        "pass_at_4": sum(any(item["reward"] for item in group["completions"]) for group in groups) / len(groups) if groups else None,
        "mixed_reward_group_rate": sum(0 < sum(item["reward"] for item in group["completions"]) < 4 for group in groups) / len(groups) if groups else None,
        "parse_rate": sum(item["parsed_answer"] is not None for item in completions) / n if n else None,
        "truncation_rate": sum(item["finish_reason"] == "length" for item in completions) / n if n else None,
    }


def save(path, manifest, groups):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"manifest": manifest, "summary": summarize(groups), "groups": groups}, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="For example: base, step4, step100")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ice_olmo_grpo.yaml")
    parser.add_argument("--model", required=True, help="Hugging Face ID or local saved-model path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--questions", type=int, default=0, help="Fixed prefix of the chosen split; 0 means the full split")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires a GPU allocation")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.is_relative_to(ROOT):
        raise ValueError("Evaluation outputs must be outside Git")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.questions < 0:
        raise ValueError("--questions must be nonnegative")
    path = output_dir / f"{args.label}-{args.split}.json"

    source_rows, all_splits, split_manifest = load_split(cfg)
    write_manifest(output_dir / "split_manifest.json", split_manifest)
    keys = all_splits[args.split][:args.questions or None]
    source = str(Path(args.model).expanduser().resolve()) if Path(args.model).expanduser().exists() else args.model
    revision = cfg["model_revision"] if source == cfg["model"] else None
    manifest = {
        "source": source, "revision": revision, "dataset": cfg["dataset"],
        "dataset_revision": cfg["dataset_revision"], "split": args.split,
        "split_sha256": split_manifest["sha256"],
        "indices": [list(key) for key in keys],
        "seed": EVAL_SEED, "group_size": 4,
        "max_new_tokens": cfg["max_completion_length"],
        "temperature": cfg["temperature"], "top_p": cfg["top_p"],
        "top_k": 0, "dtype": "bfloat16", "prompt": PROMPT,
    }
    groups = []
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["manifest"] != manifest:
            raise ValueError(f"Existing evaluation manifest differs: {path}")
        groups = previous["groups"]
    if len(groups) == len(keys):
        print(json.dumps(summarize(groups), indent=2), flush=True)
        return

    adapter = Path(source).is_dir() and (Path(source) / "adapter_config.json").is_file()
    model_source = cfg["model"] if adapter else source
    model_revision = cfg["model_revision"] if adapter else revision
    tokenizer = AutoTokenizer.from_pretrained(model_source, revision=model_revision, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_source, revision=model_revision, local_files_only=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, source, local_files_only=True)
    model = model.to("cuda").eval()
    for position in range(len(groups), len(keys)):
        part, index = keys[position]
        row = source_rows[part][index]
        truth = gsm8k_ground_truth(row["answer"])
        if truth is None:
            raise ValueError(f"Unparseable {args.split} answer at {part} row {index}")
        prompt = PROMPT.format(question=row["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        torch.manual_seed(EVAL_SEED + position)
        started = time.monotonic()
        with torch.inference_mode():
            sequences = model.generate(
                **inputs, do_sample=True, num_return_sequences=4,
                max_new_tokens=cfg["max_completion_length"],
                temperature=cfg["temperature"], top_p=cfg["top_p"], top_k=0,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = sequences[:, inputs["input_ids"].shape[1]:]
        items = []
        for tokens in generated:
            response = tokenizer.decode(tokens, skip_special_tokens=True)
            parsed = extract_final_number(response)
            items.append({
                "response": response, "parsed_answer": str(parsed) if parsed is not None else None,
                "reward": exact_number_reward(response, truth),
                "generated_tokens": len(tokens),
                "finish_reason": "length" if len(tokens) >= cfg["max_completion_length"] and tokens[-1].item() != tokenizer.eos_token_id else "eos",
            })
        groups.append({"dataset_source": part, "dataset_index": index, "truth": str(truth), "completions": items})
        save(path, manifest, groups)
        print(f"{args.label}: {position + 1}/{len(keys)}, correct={sum(item['reward'] for item in items):.0f}/4, seconds={time.monotonic() - started:.1f}", flush=True)
    print(json.dumps(summarize(groups), indent=2), flush=True)


if __name__ == "__main__":
    main()
