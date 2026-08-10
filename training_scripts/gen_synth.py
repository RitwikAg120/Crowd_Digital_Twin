"""
Generate a small synthetic 'person-blob' dataset so YOLOv8n has *some* learned
signal to detect moving blobs as 'person' (class 0), since no internet access
to pretrained COCO weights or MOT17/UCF-CC-50/Mall datasets is available in
this sandbox. This is NOT a substitute for the real fine-tuned model described
in the capstone report -- it only proves the pipeline runs end-to-end with a
genuine trained detector instead of a no-op stub.
"""
import cv2
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

W, H = 640, 640
N_TRAIN = 120
N_VAL = 20

def draw_person_blob(img, cx, cy):
    """Draws a simple humanoid blob: head circle + body ellipse."""
    body_h = random.randint(55, 90)
    body_w = random.randint(18, 30)
    head_r = body_w // 2 + random.randint(1, 4)
    color = tuple(int(c) for c in np.random.randint(40, 220, size=3))

    head_cy = cy - body_h // 2 - head_r
    cv2.circle(img, (cx, head_cy), head_r, color, -1)
    cv2.ellipse(img, (cx, cy), (body_w, body_h // 2), 0, 0, 360, color, -1)

    x1 = cx - max(body_w, head_r)
    x2 = cx + max(body_w, head_r)
    y1 = head_cy - head_r
    y2 = cy + body_h // 2
    return max(0, x1), max(0, y1), min(W, x2), min(H, y2)

def gen_image(n_people):
    img = np.random.randint(15, 45, (H, W, 3), dtype=np.uint8)  # dark background noise
    # ground texture
    cv2.rectangle(img, (0, H - 120), (W, H), (30, 30, 30), -1)
    boxes = []
    attempts = 0
    while len(boxes) < n_people and attempts < n_people * 10:
        attempts += 1
        cx = random.randint(40, W - 40)
        cy = random.randint(140, H - 60)
        x1, y1, x2, y2 = draw_person_blob(img, cx, cy)
        if (x2 - x1) < 8 or (y2 - y1) < 8:
            continue
        boxes.append((x1, y1, x2, y2))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, boxes

def to_yolo_line(box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

def build(split, n):
    img_dir = f"synth_dataset/images/{split}"
    lbl_dir = f"synth_dataset/labels/{split}"
    for i in range(n):
        n_people = random.randint(3, 14)
        img, boxes = gen_image(n_people)
        fname = f"{split}_{i:04d}"
        cv2.imwrite(f"{img_dir}/{fname}.jpg", img)
        with open(f"{lbl_dir}/{fname}.txt", "w") as f:
            f.write("\n".join(to_yolo_line(b) for b in boxes))

build("train", N_TRAIN)
build("val", N_VAL)
print(f"Generated {N_TRAIN} train + {N_VAL} val synthetic frames.")
