"""
Crowd Digital Twin — Standalone CrowdHuman Training Script (train_crowdhuman.py)
Fine-tunes YOLO11 models on converted CrowdHuman dataset.

Usage:
  python training_scripts/train_crowdhuman.py --model yolo11s.pt --epochs 35 --imgsz 640 --batch 16
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(model_name="yolo11s.pt", data_yaml="./dataset_crowdhuman/crowdhuman.yaml",
          epochs=35, imgsz=640, batch=16, device=0, name="yolo11s_crowdhuman"):
    print("=" * 70)
    print(f"       STARTING CROWDHUMAN FINE-TUNING: {model_name}")
    print("=" * 70)
    print(f"Data Config : {data_yaml}")
    print(f"Epochs      : {epochs}")
    print(f"Image Size  : {imgsz}x{imgsz}")
    print(f"Batch Size  : {batch}")
    print(f"Device      : {device}")
    print("=" * 70 + "\n")

    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        print(f"[Error] Dataset YAML file not found: {data_yaml}")
        print("Please run prepare_crowdhuman.py first: python training_scripts/prepare_crowdhuman.py --labels both")
        return None

    model = YOLO(model_name)
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=4,
        device=device,
        name=name,
        exist_ok=True
    )

    out_weights = Path(f"runs/detect/{name}/weights/best.pt")
    if out_weights.exists():
        target_pt = Path("weights/yolo11s_crowdhuman.pt")
        target_pt.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(str(out_weights), str(target_pt))
        print(f"\n[Success] Saved trained weights to: {target_pt}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 on CrowdHuman dataset")
    parser.add_argument("--model", default="yolo11s.pt", help="Base YOLO model checkpoint")
    parser.add_argument("--data", default="./dataset_crowdhuman/crowdhuman.yaml", help="Path to crowdhuman.yaml")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image resolution")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default=0, help="GPU device ID or 'cpu'")
    parser.add_argument("--name", default="yolo11s_crowdhuman", help="Run output directory name")
    args = parser.parse_args()

    train(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name
    )


if __name__ == "__main__":
    main()
