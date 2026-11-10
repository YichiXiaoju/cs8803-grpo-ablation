"""One reproducible 80/10/10 split of the pinned GSM8K train and test rows."""

import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
STUDY_INDICES = ROOT / "studies/model_selection/question_indices.json"
SPLIT_SEED = 42


def load_split(cfg):
    source = load_dataset(cfg["dataset"], cfg["dataset_config"], revision=cfg["dataset_revision"])
    study = set(json.loads(STUDY_INDICES.read_text()))
    if max(study) >= len(source["train"]):
        raise ValueError("Model-selection indices exceed GSM8K train length")
    keys = [(part, i) for part in ("train", "test") for i in range(len(source[part]))]
    n = len(keys)
    n_validation = n // 10
    n_test = n // 10
    n_train = n - n_validation - n_test
    # Previously inspected study questions stay in training, so validation/test
    # remain untouched by model selection as well as optimizer updates.
    fixed_train = [("train", i) for i in sorted(study)]
    fixed_train_set = set(fixed_train)
    remaining = [key for key in keys if key not in fixed_train_set]
    random.Random(SPLIT_SEED).shuffle(remaining)
    split = {
        "train": fixed_train + remaining[:n_train - len(fixed_train)],
        "validation": remaining[n_train - len(fixed_train):n_train - len(fixed_train) + n_validation],
        "test": remaining[n_train - len(fixed_train) + n_validation:],
    }
    manifest = {
        "dataset": cfg["dataset"], "dataset_config": cfg["dataset_config"],
        "dataset_revision": cfg["dataset_revision"], "seed": SPLIT_SEED,
        "selection_questions_forced_to_train": len(fixed_train),
        "indices": {name: [list(key) for key in values] for name, values in split.items()},
    }
    manifest["sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return source, split, manifest


def write_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError(f"Split manifest differs: {path}")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
