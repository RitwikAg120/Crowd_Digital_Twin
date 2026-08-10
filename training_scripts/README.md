# Training scripts (synthetic demo model only)

These are the scripts used to produce the bundled `weights/yolov8n_cdt.pt`
and demo videos, included for transparency/reproducibility. They are NOT
the real MOT17/UCF-CC-50/Mall training pipeline — that lives in your
`crowd_digital_twin_complete.ipynb` notebook (Cells 1-5) and requires
internet access + a GPU.

- `gen_synth.py` — generates a small synthetic "person-blob" image dataset
  (120 train / 20 val) used only because the sandbox that built this
  package had no internet access to download COCO-pretrained weights or
  real pedestrian datasets.
- `01_train_synthetic_demo_model.py` — fine-tunes YOLOv8n (built from
  architecture spec, not pretrained) on the synthetic dataset. ~3 minutes
  on a single CPU core; seconds on any GPU.
- `gen_demo_video_slow.py` — generates the bundled slow-motion synthetic
  crowd video used for low-FPS-safe ByteTrack demos.

To regenerate everything from scratch:
```bash
cd training_scripts
python gen_synth.py
python 01_train_synthetic_demo_model.py
python gen_demo_video_slow.py
# then copy runs/detect/runs_synth/cdt_synth2/weights/best.pt -> ../weights/yolov8n_cdt.pt
```
