"""Compare five ICE evaluations using the original E0 protocol and unpadded text lengths."""

import argparse
import csv
import json
from pathlib import Path

from transformers import AutoTokenizer

E0_TRAIN_TOKENS = 3_453_024
ARMS = ("E0", "E1", "E2", "E3", "E5")
FIELDS = (
    "dataset", "dataset_revision", "split", "split_sha256", "indices", "seed",
    "group_size", "max_new_tokens", "temperature", "top_p", "top_k", "dtype", "prompt",
)


def read_json(path):
    return json.loads(path.read_text())


def row_for(arm, evaluation, reference, tokenizer, training):
    manifest = evaluation["manifest"]
    if any(manifest[key] != reference[key] for key in FIELDS):
        raise ValueError(f"{arm}: evaluation protocol differs from E0")
    groups = evaluation["groups"]
    keys = [[group["dataset_source"], group["dataset_index"]] for group in groups]
    if len(groups) != 879 or keys != manifest["indices"]:
        raise ValueError(f"{arm}: expected all 879 validation questions in E0 order")
    completions = []
    for group in groups:
        if len(group["completions"]) != 4:
            raise ValueError(f"{arm}: expected four completions per question")
        completions.extend(group["completions"])
    correct = sum(item["reward"] for item in completions)
    if correct != evaluation["summary"]["correct"]:
        raise ValueError(f"{arm}: saved accuracy does not match completion records")
    text_lengths = tokenizer(
        [item["response"] for item in completions],
        add_special_tokens=False, return_attention_mask=False,
    )["input_ids"]
    return {
        "arm": arm,
        "questions": len(groups),
        "completions": len(completions),
        "correct": int(correct),
        "completion_accuracy_pct": round(100 * correct / len(completions), 4),
        "mean_answer_text_tokens": round(sum(map(len, text_lengths)) / len(completions), 4),
        "training_completion_tokens": training["attempted_completion_tokens"],
        "rejected_training_completion_tokens": training["rejected_completion_tokens"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e0", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    e0 = read_json(args.e0)
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/OLMo-2-0425-1B", revision="a1847dff35000b4271fa70afc5db10fd29fedbdf",
        local_files_only=True,
    )
    rows = []
    for arm in ARMS:
        if arm == "E0":
            evaluation = e0
            training = {"attempted_completion_tokens": E0_TRAIN_TOKENS,
                        "rejected_completion_tokens": 0}
        else:
            evaluation = read_json(args.study / "validation" / arm / f"{arm}-validation.json")
            training = read_json(args.study / "runs" / arm / "summary.json")
            if training["status"] != "budget_reached":
                raise ValueError(f"{arm}: training budget not reached")
        rows.append(row_for(arm, evaluation, e0["manifest"], tokenizer, training))
    tmp = args.output.with_suffix(".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(args.output)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
