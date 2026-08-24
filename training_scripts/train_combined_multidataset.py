"""
Crowd Digital Twin — Combined Multi-Dataset Preparation & Fine-Tuning Pipeline
Merges CrowdHuman (dense crowds) + MOT17 (pedestrian tracking) + Mall datasets
into a unified YOLO training corpus and fine-tunes YOLO11s.

Usage:
  python training_scripts/train_combined_multidataset.py --crowdhuman-dir ./CrowdHuman --mot17-dir ./MOT17 --epochs 35
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def convert_box(x, y, w, h, img_w, img_h):
    """Convert [x, y, w, h] to normalized YOLO [cx, cy, nw, nh]."""
    if w <= 0 or h <= 0:
        return None
    x = max(0.0, min(float(x), float(img_w)))
    y = max(0.0, min(float(y), float(img_h)))
    w = min(float(w), float(img_w) - x)
    h = min(float(h), float(img_h) - y)

    if w <= 0 or h <= 0:
        return None

    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return [round(cx, 6), round(cy, 6), round(nw, 6), round(nh, 6)]


def prepare_crowdhuman(data_dir, out_img_train, out_lbl_train, out_img_val, out_lbl_val):
    data_dir = Path(data_dir)
    train_odgt = data_dir / "annotation_train.odgt"
    val_odgt = data_dir / "annotation_val.odgt"
    images_dir = data_dir / "Images"

    if not train_odgt.exists():
        print(f"[Combined] CrowdHuman directory not found at: {data_dir}. Skipping CrowdHuman.")
        return 0, 0

    print(f"\n[Combined] Processing CrowdHuman Dataset from {data_dir}...")
    
    def parse_odgt(odgt_file, target_img_dir, target_lbl_dir):
        count = 0
        with open(odgt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=f"CrowdHuman {odgt_file.name}"):
            data = json.loads(line.strip())
            img_id = data["ID"]
            img_filename = f"ch_{img_id}.jpg"
            src_img_path = images_dir / f"{img_id}.jpg"

            if not src_img_path.exists():
                continue

            img = cv2.imread(str(src_img_path))
            if img is None:
                continue

            img_h, img_w = img.shape[:2]
            labels = []

            for gt in data.get("gtboxes", []):
                if gt.get("tag") != "person":
                    continue

                vbox = gt.get("vbox") or gt.get("fbox")
                hbox = gt.get("hbox")

                if vbox:
                    b_norm = convert_box(vbox[0], vbox[1], vbox[2], vbox[3], img_w, img_h)
                    if b_norm:
                        labels.append(f"0 " + " ".join(map(str, b_norm)))  # 0: person

                if hbox:
                    h_norm = convert_box(hbox[0], hbox[1], hbox[2], hbox[3], img_w, img_h)
                    if h_norm:
                        labels.append(f"1 " + " ".join(map(str, h_norm)))  # 1: head

            # Save label file
            with open(target_lbl_dir / f"ch_{img_id}.txt", "w", encoding="utf-8") as lf:
                lf.write("\n".join(labels) + "\n")

            # Copy image
            target_img = target_img_dir / img_filename
            if not target_img.exists():
                cv2.imwrite(str(target_img), img)
            count += 1
        return count

    n_train = parse_odgt(train_odgt, out_img_train, out_lbl_train)
    n_val = parse_odgt(val_odgt, out_img_val, out_lbl_val) if val_odgt.exists() else 0
    return n_train, n_val


def prepare_mot17(data_dir, out_img_train, out_lbl_train):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"

    if not train_dir.exists():
        print(f"[Combined] MOT17 dataset directory not found at: {data_dir}. Skipping MOT17.")
        return 0

    print(f"\n[Combined] Processing MOT17 Dataset from {data_dir}...")
    count = 0

    for seq in train_dir.glob("MOT17-*"):
        if not seq.is_dir() or "FRCNN" not in seq.name:
            continue

        gt_file = seq / "gt" / "gt.txt"
        img_dir = seq / "img1"

        if not gt_file.exists() or not img_dir.exists():
            continue

        frame_annotations = {}
        with open(gt_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                frame_idx = int(parts[0])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                mark = int(parts[6])
                label = int(parts[7])
                visibility = float(parts[8]) if len(parts) > 8 else 1.0

                # Label 1: Pedestrian, mark 1: active entry
                if label == 1 and mark == 1 and visibility > 0.2:
                    if frame_idx not in frame_annotations:
                        frame_annotations[frame_idx] = []
                    frame_annotations[frame_idx].append((x, y, w, h))

        for frame_idx, boxes in tqdm(frame_annotations.items(), desc=f"MOT17 {seq.name}"):
            img_file = img_dir / f"{frame_idx:06d}.jpg"
            if not img_file.exists():
                continue

            img = cv2.imread(str(img_file))
            if img is None:
                continue

            img_h, img_w = img.shape[:2]
            labels = []

            for x, y, w, h in boxes:
                b_norm = convert_box(x, y, w, h, img_w, img_h)
                if b_norm:
                    labels.append(f"0 " + " ".join(map(str, b_norm)))

            frame_id = f"mot17_{seq.name}_{frame_idx:06d}"
            with open(out_lbl_train / f"{frame_id}.txt", "w", encoding="utf-8") as lf:
                lf.write("\n".join(labels) + "\n")

            cv2.imwrite(str(out_img_train / f"{frame_id}.jpg"), img)
            count += 1

    return count


def create_combined_yaml(out_dir):
    out_path = Path(out_dir).absolute()
    yaml_path = out_path / "combined_crowd.yaml"
    content = f"""path: {out_path.as_posix()}
train: images/train
val: images/val

names:
  0: person
  1: head
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[Combined] Generated unified dataset config at: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Combined Multi-Dataset Training Pipeline for Crowd Digital Twin")
    parser.add_argument("--crowdhuman-dir", default="./CrowdHuman", help="Path to CrowdHuman root directory")
    parser.add_argument("--mot17-dir", default="./MOT17", help="Path to MOT17 root directory")
    parser.add_argument("--out-dir", default="./dataset_combined", help="Target directory for combined YOLO dataset")
    parser.add_argument("--model", default="yolo11s.pt", help="Base YOLO checkpoint")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image resolution")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default=0, help="GPU device ID or 'cpu'")
    parser.add_argument("--skip-prep", action="store_true", help="Skip dataset preparation if already prepared")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_img_train = out_dir / "images" / "train"
    out_lbl_train = out_dir / "labels" / "train"
    out_img_val = out_dir / "images" / "val"
    out_lbl_val = out_dir / "labels" / "val"

    if not args.skip_prep:
        out_img_train.mkdir(parents=True, exist_ok=True)
        out_lbl_train.mkdir(parents=True, exist_ok=True)
        out_img_val.mkdir(parents=True, exist_ok=True)
        out_lbl_val.mkdir(parents=True, exist_ok=True)

        n_ch_tr, n_ch_val = prepare_crowdhuman(args.crowdhuman_dir, out_img_train, out_lbl_train, out_img_val, out_lbl_val)
        n_mot_tr = prepare_mot17(args.mot17_dir, out_img_train, out_lbl_train)

        print("\n" + "=" * 60)
        print("         COMBINED DATASET SUMMARY")
        print("=" * 60)
        print(f"CrowdHuman Train Images : {n_ch_tr}")
        print(f"CrowdHuman Val Images   : {n_ch_val}")
        print(f"MOT17 Train Frames      : {n_mot_tr}")
        print(f"Total Merged Train Set  : {n_ch_tr + n_mot_tr}")
        print("=" * 60 + "\n")

    yaml_path = create_combined_yaml(out_dir)

    print("\n" + "=" * 60)
    print(f"  STARTING MULTI-DATASET FINE-TUNING ({args.model})")
    print("=" * 60)
    model = YOLO(args.model)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=4,
        device=args.device,
        name="yolo11s_combined_cdt",
        exist_ok=True
    )

    out_pt = Path("runs/detect/yolo11s_combined_cdt/weights/best.pt")
    if out_pt.exists():
        target_pt = Path("weights/yolo11s_crowdhuman.pt")
        target_pt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(out_pt), str(target_pt))
        print(f"\n[Success] Unified multi-dataset weights saved to: {target_pt}")


if __name__ == "__main__":
    main()
