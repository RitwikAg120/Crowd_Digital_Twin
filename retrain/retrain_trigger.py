"""Retrain trigger skeleton.

Scans the `experience/logs` directory for accumulated samples and optionally
invokes the offline training script when a threshold is met. This is a
lightweight orchestrator — extend to integrate with CI, MLFlow, or a model
registry for production use.

Usage:
  python retrain_trigger.py --min-samples 200 --run
"""
import argparse
import json
import os
import subprocess
from pathlib import Path


def count_samples(logs_dir: Path) -> int:
    if not logs_dir.exists():
        return 0
    total = 0
    for f in logs_dir.glob("*.jsonl"):
        try:
            with open(f, "r", encoding="utf8") as fh:
                for _ in fh:
                    total += 1
        except Exception:
            continue
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-samples", type=int, default=200)
    p.add_argument("--run", action="store_true",
                   help="Actually run the training script when threshold met")
    p.add_argument("--logs", default="experience/logs")
    args = p.parse_args()

    logs = Path(args.logs)
    total = count_samples(logs)
    print(f"Found {total} experience samples in {logs}")

    if total < args.min_samples:
        print(f"Not enough samples (need {args.min_samples}). Exiting.")
        return

    print("Threshold met — recommended action: kick off offline training.")
    if args.run:
        print("Running training script (this may be slow).")
        # Call the existing training script. Adjust path if you relocate files.
        script = Path(__file__).resolve().parents[1] / "training_scripts" / "01_train_synthetic_demo_model.py"
        if script.exists():
            subprocess.run(["python", str(script)], check=False)
        else:
            print(f"Training script not found at {script}. Start training manually.")


if __name__ == "__main__":
    main()
