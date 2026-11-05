"""Validate full model-selection audits and write a compact result artifact."""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/model_selection"
RUNS = ROOT / "runs/model_selection"


def main():
    config = yaml.safe_load((STUDY / "config.yaml").read_text())
    indices = json.loads((STUDY / "question_indices.json").read_text())
    if len(indices) != config["num_prompts"] or len(set(indices)) != len(indices):
        raise RuntimeError("Question indices are missing or duplicated")
    results = {}
    for name, spec in config["models"].items():
        data = json.loads((RUNS / f"{name}.json").read_text())
        groups = data["groups"]
        if [g["dataset_index"] for g in groups] != indices:
            raise RuntimeError(f"{name}: question order differs")
        if any(len(g["completions"]) != config["group_size"] for g in groups):
            raise RuntimeError(f"{name}: group size differs")
        items = [item for group in groups for item in group["completions"]]
        expected_seeds = [
            config["seed"] + i * config["group_size"] + j
            for i in range(config["num_prompts"])
            for j in range(config["group_size"])
        ]
        if [item["seed"] for item in items] != expected_seeds:
            raise RuntimeError(f"{name}: sample seeds differ")
        s = data["summary"]
        if s["model"] != spec["id"] or s["revision"] != spec["revision"]:
            raise RuntimeError(f"{name}: checkpoint differs")
        n = len(items)
        mixed = sum(0 < sum(x["reward"] for x in g["completions"]) < config["group_size"] for g in groups)
        result = {
            "model": spec["id"], "revision": spec["revision"],
            "backend": spec["backend"], "precision": spec["precision"],
            "prompt_count": len(groups), "completion_count": n,
            "correct_completions": int(sum(x["reward"] for x in items)),
            "completion_accuracy": sum(x["reward"] for x in items) / n,
            "mixed_reward_groups": mixed,
            "mixed_reward_group_rate": mixed / len(groups),
            "parsed_completions": sum(x["parsed_answer"] is not None for x in items),
            "truncated_completions": sum(x["finish_reason"] == "length" for x in items),
            "generated_tokens": sum(x["generated_tokens"] for x in items),
            "generation_seconds": round(sum(x["generation_seconds"] for x in items), 2),
        }
        if result["completion_accuracy"] != s["completion_accuracy"] or result["mixed_reward_group_rate"] != s["mixed_reward_group_rate"]:
            raise RuntimeError(f"{name}: summary metrics differ from raw records")
        results[name] = result
    artifact = {
        "dataset": config["dataset"], "dataset_config": config["dataset_config"],
        "dataset_revision": config["dataset_revision"], "split": config["split"],
        "seed": config["seed"], "num_prompts": config["num_prompts"],
        "group_size": config["group_size"],
        "temperature": config["temperature"], "top_p": config["top_p"],
        "max_completion_length": config["max_completion_length"],
        "training_performed": False, "results": results,
    }
    path = STUDY / "results.json"
    path.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"Validated all {len(results)} 100 x 4 audits and wrote {path}")


if __name__ == "__main__":
    main()
