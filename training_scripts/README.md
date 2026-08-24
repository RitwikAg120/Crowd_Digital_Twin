# Training & Evaluation Scripts

This directory contains data preparation, fine-tuning, quantitative evaluation, and ablation benchmarking scripts for the **Crowd Digital Twin (CDT)**.

---

## 1. CrowdHuman Dataset Fine-Tuning (Real Pedestrians & Heads)

### Step 1: Dataset Preparation
Converts CrowdHuman `.odgt` annotations to YOLO format supporting both full body (`person`) and visible head (`head`) labels:
```bash
python training_scripts/prepare_crowdhuman.py --data-dir ./CrowdHuman --out-dir ./dataset_crowdhuman --labels both
```

### Step 2: Model Training (CLI / Kaggle / Colab GPU)
Train YOLO11s / YOLO11n on the converted CrowdHuman dataset:
```bash
# Standalone CLI training script
python training_scripts/train_crowdhuman.py --model yolo11s.pt --epochs 35 --imgsz 640 --batch 16 --device 0

# Notebook alternative for Kaggle / Colab GPU:
# See reference/train_kaggle_crowdhuman.ipynb
```
Trained best weights are automatically saved to `weights/yolo11s_crowdhuman.pt`.

---

## 2. Quantitative Evaluation & SAHI Ablation

### Model Evaluation Matrix (`eval_count.py`)
Calculates Mean Absolute Error (MAE), RMSE, and FPS across model sizes and input resolutions ($640 \times 640$ vs $960 \times 960$):
```bash
python training_scripts/eval_count.py --dataset ./dataset_crowdhuman
```

### SAHI & NMS-Free Ablation (`eval_ablation_sahi_yolo26.py`)
Compares Standard YOLO11s vs SAHI Sliced Inference ($320 \times 320$ slices) vs NMS-Free End-to-End architectures on dense crowds:
```bash
python training_scripts/eval_ablation_sahi_yolo26.py --image bus.jpg
```

---

## 3. Repository Workflow Summary

- Data Preparation: [`prepare_crowdhuman.py`](file:///c:/Users/rishi/OneDrive/Desktop/Github%20projects/CrowdDT/training_scripts/prepare_crowdhuman.py)
- Standalone Fine-Tuning: [`train_crowdhuman.py`](file:///c:/Users/rishi/OneDrive/Desktop/Github%20projects/CrowdDT/training_scripts/train_crowdhuman.py)
- Unified Multi-Dataset Fine-Tuning: [`train_combined_multidataset.py`](file:///c:/Users/rishi/OneDrive/Desktop/Github%20projects/CrowdDT/training_scripts/train_combined_multidataset.py)
- Quantitative Evaluation: [`eval_count.py`](file:///c:/Users/rishi/OneDrive/Desktop/Github%20projects/CrowdDT/training_scripts/eval_count.py)
- SAHI & NMS-Free Ablation: [`eval_ablation_sahi_yolo26.py`](file:///c:/Users/rishi/OneDrive/Desktop/Github%20projects/CrowdDT/training_scripts/eval_ablation_sahi_yolo26.py)

