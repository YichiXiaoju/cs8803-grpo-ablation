"""Report optimizer steps completed by a resumable GRPO training run."""

import argparse
import json
import time
from pathlib import Path
from typing import Optional


def latest_complete_step(trainer_dir: Path) -> Optional[int]:
    """Return the largest step whose checkpoint has finished writing."""
    candidates = []
    for checkpoint in trainer_dir.glob("checkpoint-*"):
        try:
            step = int(checkpoint.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        state = checkpoint / "trainer_state.json"
        required = checkpoint / "optimizer.pt", checkpoint / "scheduler.pt"
        if not state.is_file() or not all(path.is_file() for path in required):
            continue
        if (checkpoint / "model.safetensors").is_file() or (
            checkpoint / "adapter_model.safetensors"
        ).is_file():
            candidates.append(step)
            continue
        index = checkpoint / "model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            if all((checkpoint / shard).is_file() for shard in set(weight_map.values())):
                candidates.append(step)
    return max(candidates) if candidates else None


def report(output: Path) -> bool:
    """Print progress and return whether the configured target has been reached."""
    manifest_path = output / "manifest.json"
    target = None
    if manifest_path.is_file():
        target = json.loads(manifest_path.read_text())["config"].get("max_steps")

    step = latest_complete_step(output / "trainer")
    if step is None:
        message = "No complete checkpoint yet"
        reached = False
    else:
        suffix = f"/{target}" if target is not None else ""
        message = f"Completed optimizer steps: {step}{suffix}"
        reached = target is not None and step >= target
    print(message, flush=True)
    return reached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watch", type=float, metavar="SECONDS", help="Poll until training finishes")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()

    if args.watch is None:
        report(output)
        return
    if args.watch <= 0:
        raise ValueError("--watch must be positive")
    while not report(output):
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
