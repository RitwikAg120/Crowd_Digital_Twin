"""
CrowdHuman to YOLO Format Converter
Supports full body, visible body, and head bounding box labels.

Usage:
  python prepare_crowdhuman.py --data-dir ./CrowdHuman --out-dir ./dataset_crowdhuman --labels both
"""

import argparse
import json
import os
from pathlib import Path
import cv2
from tqdm import tqdm


def convert_box(box, img_w, img_h):
    """Convert [x, y, w, h] to normalized [cx, cy, nw, nh]."""
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return None
    # Clip coordinates
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


def process_odgt(odgt_path, images_dir, out_img_dir, out_lbl_dir, labels_mode="both"):
    odgt_path = Path(odgt_path)
    if not odgt_path.exists():
        print(f"[Warning] ODGT file not found: {odgt_path}")
        return 0

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(odgt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=f"Processing {odgt_path.name}"):
        data = json.loads(line.strip())
        img_id = data["ID"]
        img_filename = f"{img_id}.jpg"
        img_path = Path(images_dir) / img_filename

        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        yolo_labels = []

        for gt in data.get("gtboxes", []):
            if gt.get("tag") != "person":
                continue

            # Body box (vbox preferred, fallback to fbox)
            vbox = gt.get("vbox") or gt.get("fbox")
            hbox = gt.get("hbox")

            if labels_mode in ("both", "body") and vbox:
                box_norm = convert_box(vbox, img_w, img_h)
                if box_norm:
                    class_id = 0  # 0: person/body
                    yolo_labels.append(f"{class_id} " + " ".join(map(str, box_norm)))

            if labels_mode in ("both", "head") and hbox:
                box_norm = convert_box(hbox, img_w, img_h)
                if box_norm:
                    class_id = 1 if labels_mode == "both" else 0  # 1: head if both, 0 if head-only
                    yolo_labels.append(f"{class_id} " + " ".join(map(str, box_norm)))

        # Write label file
        label_path = out_lbl_dir / f"{img_id}.txt"
        with open(label_path, "w", encoding="utf-8") as lf:
            lf.write("\n".join(yolo_labels) + "\n")

        # Copy/link image reference file
        target_img_path = out_img_dir / img_filename
        if not target_img_path.exists():
            try:
                os.link(str(img_path), str(target_img_path))
            except Exception:
                cv2.imwrite(str(target_img_path), img)

        count += 1

    return count


def create_yaml_config(out_dir, labels_mode="both"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    yaml_path = out_path / "crowdhuman.yaml"

    if labels_mode == "both":
        names = {0: "person", 1: "head"}
    elif labels_mode == "head":
        names = {0: "head"}
    else:
        names = {0: "person"}

    content = f"""path: {out_path.absolute().as_posix()}
train: images/train
val: images/val

names:
"""
    for cid, cname in names.items():
        content += f"  {cid}: {cname}\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[Dataset] Generated YOLO YAML config at: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert CrowdHuman dataset to YOLO format")
    parser.add_argument("--data-dir", "--crowdhuman-dir", dest="data_dir", default="./CrowdHuman", help="Root CrowdHuman data directory")
    parser.add_argument("--out-dir", default="./dataset_crowdhuman", help="Output YOLO dataset directory")
    parser.add_argument("--labels", choices=["both", "body", "head"], default="both", help="Label types to export")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Converting CrowdHuman Dataset (Labels: {args.labels}) ===")
    
    train_odgt = data_dir / "annotation_train.odgt"
    val_odgt = data_dir / "annotation_val.odgt"

    if not train_odgt.exists() and not val_odgt.exists():
        print(f"\n[Notice] CrowdHuman dataset annotations not found in '{data_dir.absolute()}'.")
        print("To download the dataset:")
        print("  1. Download CrowdHuman from http://www.crowdhuman.org/ or Kaggle ('crowdhuman-dataset')")
        print("  2. Place 'annotation_train.odgt', 'annotation_val.odgt', and 'Images/' inside './CrowdHuman/'")
        create_yaml_config(out_dir, args.labels)
        print(f"[Notice] Created empty target structure and config at {out_dir.absolute()}\n")
        return

    val_img_dir = data_dir / "Images_val" if (data_dir / "Images_val").exists() else data_dir / "Images"
    n_train = process_odgt(
        odgt_path=train_odgt,
        images_dir=data_dir / "Images",
        out_img_dir=out_dir / "images" / "train",
        out_lbl_dir=out_dir / "labels" / "train",
        labels_mode=args.labels
    )
    n_val = process_odgt(
        odgt_path=val_odgt,
        images_dir=val_img_dir,
        out_img_dir=out_dir / "images" / "val",
        out_lbl_dir=out_dir / "labels" / "val",
        labels_mode=args.labels
    )

    create_yaml_config(out_dir, args.labels)
    print(f"=== Conversion Complete! Train images: {n_train}, Val images: {n_val} ===")


if __name__ == "__main__":
    main()
