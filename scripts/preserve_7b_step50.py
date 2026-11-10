"""Keep an inference-ready copy of the 7B model at optimizer step 50.

Run alongside training. Hard links avoid copying 28 GB while the checkpoint
rotation is in progress; the saved weights survive deletion of checkpoint-50.
"""

import json
import os
import time
from pathlib import Path


def main():
    root = Path.home() / "scratch/grpo/runs/olmo2-7b-full-grpo-1epoch-lr1e-6"
    trainer = root / "trainer"
    checkpoint = trainer / "checkpoint-50"
    destination = root / "milestones/step-50-model"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (destination / "COMPLETE").exists():
        print(f"Already preserved: {destination}", flush=True)
        return

    while True:
        state_file = checkpoint / "trainer_state.json"
        index_file = checkpoint / "model.safetensors.index.json"
        optimizer_file = checkpoint / "optimizer.bin"
        if state_file.exists() and index_file.exists() and optimizer_file.exists():
            try:
                step = json.loads(state_file.read_text())["global_step"]
                shard_names = sorted(set(json.loads(index_file.read_text())["weight_map"].values()))
                complete = (
                    step == 50
                    and len(shard_names) > 0
                    and optimizer_file.stat().st_size > 50_000_000_000
                    and (checkpoint / "scheduler.pt").exists()
                    and not (trainer / "checkpoint-45").exists()
                    and all((checkpoint / name).exists() for name in shard_names)
                )
            except (OSError, ValueError, KeyError):
                complete = False
            if complete:
                destination.mkdir(exist_ok=True)
                for name in shard_names:
                    target = destination / name
                    if not target.exists():
                        os.link(checkpoint / name, target)
                for name in (
                    "config.json", "generation_config.json", "merges.txt",
                    "model.safetensors.index.json", "special_tokens_map.json",
                    "tokenizer.json", "tokenizer_config.json", "vocab.json",
                ):
                    source = checkpoint / name
                    if source.exists():
                        target = destination / name
                        if not target.exists():
                            os.link(source, target)
                (destination / "COMPLETE").write_text(
                    json.dumps({"step": 50, "source": str(checkpoint), "shards": shard_names}, indent=2) + "\n"
                )
                print(f"Preserved step-100 model at {destination}", flush=True)
                return
        time.sleep(5)


if __name__ == "__main__":
    main()
