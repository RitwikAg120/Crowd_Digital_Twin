"""
Crowd Digital Twin — Model Evaluation Script (eval_count.py)
Evaluates object detection models on crowd datasets (CrowdHuman val, MOT17, or custom frames)
and builds quantitative comparison tables: COCO-base vs fine-tuned x (n vs s) x (640 vs 960).
"""

import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


def evaluate_model_on_dataset(model_path, dataset_dir, conf_thr=0.35, imgsz=640):
    model_name = Path(model_path).stem
    print(f"\nEvaluating Model: {model_name} (Resolution: {imgsz}x{imgsz}, Conf: {conf_thr})...", flush=True)

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        return None

    dataset_path = Path(dataset_dir)
    image_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
    if not image_files:
        # Fallback to images subdirectory if available
        image_files = list(dataset_path.glob("images/**/*.jpg")) + list(dataset_path.glob("images/**/*.png"))

    if not image_files:
        print(f"No images found in {dataset_dir}")
        return None

    counts = []
    latencies = []
    gt_counts = []  # Read GT label files if present in labels/

    for img_path in image_files[:100]:  # Evaluate on up to 100 samples
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Ground truth check
        lbl_path = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                gt_c = sum(1 for line in f if line.strip().startswith("0"))
            gt_counts.append(gt_c)

        t0 = time.perf_counter()
        res = model(img, classes=[0], conf=conf_thr, imgsz=imgsz, verbose=False)
        t1 = time.perf_counter()

        dets = sv.Detections.from_ultralytics(res[0])
        counts.append(len(dets))
        latencies.append((t1 - t0) * 1000.0)

    mean_count = float(np.mean(counts)) if counts else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    fps = 1000.0 / max(mean_lat, 0.1)

    mae, rmse = 0.0, 0.0
    if len(gt_counts) == len(counts) and len(counts) > 0:
        errors = np.abs(np.array(counts) - np.array(gt_counts))
        mae = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))

    return {
        "model": model_name,
        "imgsz": imgsz,
        "n_samples": len(counts),
        "mean_count": round(mean_count, 2),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "fps": round(fps, 1),
        "latency_ms": round(mean_lat, 1),
    }


def print_ablation_table(results):
    print("\n" + "=" * 80)
    print("                      CROWD COUNTING EVALUATION TABLE")
    print("=" * 80)
    header = f"| {'Model':<20} | {'Resolution':<10} | {'Mean Count':<12} | {'MAE':<8} | {'RMSE':<8} | {'FPS':<6} | {'Latency':<10} |"
    divider = "|-" + "-|-".join(["-"*20, "-"*10, "-"*12, "-"*8, "-"*8, "-"*6, "-"*10]) + "-|"
    print(header)
    print(divider)

    for r in results:
        if r is None: continue
        print(f"| {r['model']:<20} | {r['imgsz']:<10} | {r['mean_count']:<12.2f} | {r['mae']:<8.2f} | {r['rmse']:<8.2f} | {r['fps']:<6.1f} | {r['latency_ms']:<7.1f} ms |")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate detection models on crowd counting benchmarks")
    parser.add_argument("--dataset", default="./videos", help="Directory containing dataset images or videos")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()

    models_to_eval = [
        ("yolov8n.pt", 640),
        ("yolo11n.pt", 640),
        ("yolo11s.pt", 640),
        ("yolo11s.pt", 960),
    ]

    # Include custom weights if present
    custom_weights = Path("weights/yolo11s_crowdhuman.pt")
    if custom_weights.exists():
        models_to_eval.append((str(custom_weights), 640))
        models_to_eval.append((str(custom_weights), 960))

    results = []
    for mpath, sz in models_to_eval:
        res = evaluate_model_on_dataset(mpath, args.dataset, conf_thr=args.conf, imgsz=sz)
        if res:
            results.append(res)

    print_ablation_table(results)


if __name__ == "__main__":
    main()
