"""Retrain trigger.

Counts the samples accumulated in `experience/logs` and reports whether the
threshold for a retraining run has been met. Training itself is done offline
with the notebooks in `reference/` (yolo26smodel.ipynb, yoloheadv1.ipynb).

Usage:
  python retrain/retrain_trigger.py --min-samples 200
"""
import argparse
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
    p.add_argument("--logs", default="experience/logs")
    args = p.parse_args()

    logs = Path(args.logs)
    total = count_samples(logs)
    print(f"Found {total} experience samples in {logs}")

    if total < args.min_samples:
        print(f"Not enough samples (need {args.min_samples}). Exiting.")
        return

    print("Threshold met — retrain offline with the notebooks in reference/.")


if __name__ == "__main__":
    main()
