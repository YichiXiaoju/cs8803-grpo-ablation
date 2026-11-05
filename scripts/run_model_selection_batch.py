"""Run model selection models sequentially, resuming saved questions."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/model_selection"


def write_status(model, state, **extra):
    payload = {"model": model, "state": state, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra}
    pending = OUT / "batch_status.tmp"
    pending.write_text(json.dumps(payload, indent=2) + "\n")
    pending.replace(OUT / "batch_status.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=("qwen", "qwen_1_5b", "gemma", "smollm", "olmo"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        write_status(model, "running")
        with (OUT / f"{model}.log").open("a") as log:
            print(f"\n=== Started {model} at {datetime.now(timezone.utc).isoformat()} ===", file=log, flush=True)
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/model_selection.py"), "--model", model],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False,
            )
            print(f"=== Finished {model}: exit {result.returncode} ===", file=log, flush=True)
        if result.returncode:
            write_status(model, "failed", exit_code=result.returncode)
            return result.returncode
    write_status(args.models[-1], "completed", completed_models=args.models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
