"""Inference-only, resumable 100 x 4 GSM8K checkpoint comparison on a Mac."""

import argparse
import json
import os
import re
import time
from pathlib import Path

import yaml
from datasets import load_dataset

from grpo_ablation.rewards import exact_number_reward, extract_final_number, gsm8k_ground_truth

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "studies/model_selection/config.yaml"
INDICES = ROOT / "studies/model_selection/question_indices.json"
PROMPT = (
    "Solve the math problem step by step. End with a separate line in the "
    "format Final answer: <number>.\n\n"
    "Question: Ana has 2 pens and buys 3 more. How many pens does she have?\n"
    "Solution: 2 + 3 = 5.\nFinal answer: 5\n\n"
    "Question: {question}\nSolution:"
)
ANSWER_LINE = re.compile(
    r"(?im)^final\s+answer\s*:\s*[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?[ \t]*[.!]?[ \t]*\r?\n"
)


def fixed_indices(data, cfg):
    import random

    expected = random.Random(cfg["seed"]).sample(range(len(data)), cfg["num_prompts"])
    if INDICES.exists():
        saved = json.loads(INDICES.read_text())
        if saved != expected:
            raise RuntimeError("Saved question indices differ from dataset/config; stop and inspect")
    else:
        INDICES.write_text(json.dumps(expected, indent=2) + "\n")
    return expected


def load_backend(spec):
    started = time.monotonic()
    if spec["backend"] == "mlx":
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = load(spec["id"], revision=spec["revision"])

        def generate(prompt, seed, cfg):
            mx.random.seed(seed)
            sampler = make_sampler(temp=cfg["temperature"], top_p=cfg["top_p"])
            parts = []
            last = None
            match = None
            for item in stream_generate(
                model, tokenizer, prompt, max_tokens=cfg["max_completion_length"], sampler=sampler
            ):
                parts.append(item.text)
                last = item
                match = ANSWER_LINE.search("".join(parts))
                if match:
                    break
            if last is None:
                raise RuntimeError("MLX generated no output")
            response = "".join(parts)
            return (
                response[:match.end()] if match else response,
                last.generation_tokens,
                "answer_marker" if match else last.finish_reason,
            )

    elif spec["backend"] == "mps":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

        if not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS is unavailable")
        tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
        model = AutoModelForCausalLM.from_pretrained(
            spec["id"], revision=spec["revision"], dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to("mps").eval()

        class StopAtAnswer(StoppingCriteria):
            def __init__(self, prompt_tokens):
                self.prompt_tokens = prompt_tokens

            def __call__(self, input_ids, scores, **kwargs):
                text = tokenizer.decode(input_ids[0, self.prompt_tokens:], skip_special_tokens=True)
                return bool(ANSWER_LINE.search(text))

        def generate(prompt, seed, cfg):
            torch.manual_seed(seed)
            inputs = tokenizer(prompt, return_tensors="pt").to("mps")
            prompt_tokens = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                output = model.generate(
                    **inputs, do_sample=True, temperature=cfg["temperature"],
                    top_p=cfg["top_p"], top_k=0,
                    max_new_tokens=cfg["max_completion_length"],
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([StopAtAnswer(prompt_tokens)]),
                )
            tokens = output[0, prompt_tokens:]
            response = tokenizer.decode(tokens, skip_special_tokens=True)
            match = ANSWER_LINE.search(response)
            finish = (
                "answer_marker" if match else
                "eos" if len(tokens) and tokens[-1].item() == tokenizer.eos_token_id else
                "length" if len(tokens) >= cfg["max_completion_length"] else "other"
            )
            return response[:match.end()] if match else response, len(tokens), finish

    else:
        raise ValueError(f"Unsupported backend: {spec['backend']}")
    return generate, round(time.monotonic() - started, 2)


def summary(groups, cfg, spec, load_seconds, elapsed_seconds):
    completions = [item for group in groups for item in group["completions"]]
    n = len(completions)
    return {
        "model": spec["id"], "revision": spec["revision"],
        "backend": spec["backend"], "precision": spec["precision"],
        "prompt_count": len(groups), "completion_count": n,
        "group_size": cfg["group_size"], "training_performed": False,
        "completion_accuracy": sum(x["reward"] for x in completions) / n if n else None,
        "mixed_reward_group_rate": sum(
            0 < sum(x["reward"] for x in g["completions"]) < cfg["group_size"]
            for g in groups
        ) / len(groups) if groups else None,
        "parse_rate": sum(x["parsed_answer"] is not None for x in completions) / n if n else None,
        "truncation_rate": sum(x["finish_reason"] == "length" for x in completions) / n if n else None,
        "generated_tokens": sum(x["generated_tokens"] for x in completions),
        "load_seconds": load_seconds,
        "generation_seconds": round(sum(x["generation_seconds"] for x in completions), 2),
        "elapsed_seconds_this_invocation": round(elapsed_seconds, 2),
    }


def save(path, groups, cfg, spec, load_seconds, started):
    payload = {"summary": summary(groups, cfg, spec, load_seconds, time.monotonic() - started), "groups": groups}
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-prompts", type=int, default=None, help="Smoke test limit; default is 100")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())
    if args.model not in cfg["models"]:
        parser.error(f"Unknown model {args.model!r}; choose from {', '.join(cfg['models'])}")
    spec = cfg["models"][args.model]
    count = args.num_prompts if args.num_prompts is not None else cfg["num_prompts"]
    if not 1 <= count <= cfg["num_prompts"]:
        raise ValueError("num-prompts must be between 1 and the fixed study size")
    for key in ("HF_HOME", "HF_HUB_CACHE"):
        if key in os.environ and Path(os.environ[key]).expanduser().resolve().is_relative_to(ROOT):
            raise RuntimeError(f"{key} must be outside the repository")
    output = ROOT / "runs/model_selection" / f"{args.model}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"Loading dataset and {spec['id']}...", flush=True)
    data = load_dataset(
        cfg["dataset"], cfg["dataset_config"], split=cfg["split"],
        revision=cfg["dataset_revision"],
    )
    indices = fixed_indices(data, cfg)
    groups = json.loads(output.read_text())["groups"] if output.exists() else []
    if len(groups) > count:
        raise RuntimeError("Existing output has more prompts than requested; use the full run or inspect")
    for i, group in enumerate(groups):
        if group["dataset_index"] != indices[i] or len(group["completions"]) != cfg["group_size"]:
            raise RuntimeError("Existing output is incompatible with fixed study")
    if len(groups) == count:
        print(f"Already complete: {output}", flush=True)
        return
    generate, load_seconds = load_backend(spec)
    print(f"Loaded in {load_seconds}s; resuming at question {len(groups) + 1}/{count}", flush=True)
    for i in range(len(groups), count):
        row = data[indices[i]]
        truth = gsm8k_ground_truth(row["answer"])
        if truth is None:
            raise RuntimeError(f"Cannot parse ground truth for dataset index {indices[i]}")
        prompt = PROMPT.format(question=row["question"])
        group = {"dataset_index": indices[i], "question": row["question"], "truth": str(truth), "completions": []}
        for j in range(cfg["group_size"]):
            seed = cfg["seed"] + i * cfg["group_size"] + j
            t0 = time.monotonic()
            response, tokens, finish = generate(prompt, seed, cfg)
            parsed = extract_final_number(response)
            item = {
                "seed": seed, "response": response,
                "parsed_answer": str(parsed) if parsed is not None else None,
                "reward": exact_number_reward(response, truth),
                "generated_tokens": tokens, "finish_reason": finish,
                "generation_seconds": round(time.monotonic() - t0, 2),
            }
            group["completions"].append(item)
            print(f"{args.model} question {i+1}/{count} sample {j+1}/{cfg['group_size']}: reward={item['reward']:.0f}, tokens={tokens}, seconds={item['generation_seconds']}", flush=True)
        groups.append(group)
        save(output, groups, cfg, spec, load_seconds, started)
    print(json.dumps(summary(groups, cfg, spec, load_seconds, time.monotonic() - started), indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
