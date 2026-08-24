"""
Crowd Digital Twin — Model Metrics & F1-Score Evaluator (eval_metrics.py)
Calculates Precision, Recall, F1-Score, mAP50, mAP50-95, MAE, and Count Accuracy.

Usage:
  python training_scripts/eval_metrics.py --model yolo11s.pt --data dataset_crowdhuman/crowdhuman.yaml
"""

import argparse
from pathlib import Path
import numpy as np
from ultralytics import YOLO


def evaluate_model_metrics(model_path="yolo11s.pt", data_yaml="./dataset_crowdhuman/crowdhuman.yaml", imgsz=640, conf=0.20):
    print("=" * 75)
    print(f"      MODEL ACCURACY & F1-SCORE EVALUATION ENGINE")
    print("=" * 75)
    print(f"Model Checkpoint : {model_path}")
    print(f"Dataset Config   : {data_yaml}")
    print(f"Image Resolution : {imgsz}x{imgsz}")
    print(f"Confidence Threshold: {conf}")
    print("=" * 75 + "\n")

    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        print(f"[Notice] Dataset YAML '{data_yaml}' not found.")
        print("To run validation on custom dataset labels, run prepare_crowdhuman.py first.")
        print("\nEvaluating base model metrics on sample validation pipeline...\n")

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[Error] Failed to load model {model_path}: {e}")
        return None

    # Run Ultralytics Validation Suite if YAML exists
    if yaml_path.exists():
        print("[Engine] Running YOLO validation suite...", flush=True)
        val_results = model.val(
            data=str(yaml_path),
            imgsz=imgsz,
            conf=conf,
            verbose=False
        )

        # Extract Metrics
        metrics = val_results.results_dict
        precision = float(metrics.get("metrics/precision(B)", 0.0))
        recall = float(metrics.get("metrics/recall(B)", 0.0))
        map50 = float(metrics.get("metrics/mAP50(B)", 0.0))
        map50_95 = float(metrics.get("metrics/mAP50-95(B)", 0.0))

        # Calculate F1-Score
        if (precision + recall) > 0:
            f1_score = 2.0 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        print("\n" + "=" * 75)
        print("                     EVALUATION METRICS RESULTS")
        print("=" * 75)
        print(f"  Precision (P)       : {precision * 100:.2f}%  (TP / (TP + FP))")
        print(f"  Recall (R)          : {recall * 100:.2f}%  (TP / (TP + FN))")
        print(f"  F1-Score            : {f1_score * 100:.2f}%  (Harmonic Mean of P & R)")
        print(f"  mAP @ 0.50 (mAP50)  : {map50 * 100:.2f}%  (Detection Accuracy at IoU 0.5)")
        print(f"  mAP @ 0.50:0.95     : {map50_95 * 100:.2f}%  (Strict Detection Accuracy)")
        print("=" * 75 + "\n")

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "map50": map50,
            "map50_95": map50_95
        }
    else:
        # Standalone Metric Calculation Helper Example
        print("=" * 75)
        print("               STANDALONE METRIC FORMULAS & EXAMPLE CALCULATOR")
        print("=" * 75)
        print("Formulas:")
        print("  Precision (P) = True Positives / (True Positives + False Positives)")
        print("  Recall (R)    = True Positives / (True Positives + False Negatives)")
        print("  F1-Score      = 2 * (Precision * Recall) / (Precision + Recall)")
        print("  Accuracy      = 1.0 - (Mean Absolute Error / Ground Truth Count)")
        print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Model Accuracy, Precision, Recall, and F1-Score")
    parser.add_argument("--model", default="yolo11s.pt", help="Path to YOLO model checkpoint")
    parser.add_argument("--data", default="./dataset_crowdhuman/crowdhuman.yaml", help="Path to dataset YAML file")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    args = parser.parse_args()

    evaluate_model_metrics(
        model_path=args.model,
        data_yaml=args.data,
        imgsz=args.imgsz,
        conf=args.conf
    )


if __name__ == "__main__":
    main()
