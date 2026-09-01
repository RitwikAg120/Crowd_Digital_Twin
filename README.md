# Crowd Digital Twin (CDT) 



## 1. What's included

| File / folder              | What it is                                                              |
|-----------------------------|--------------------------------------------------------------------------|
| `main.py`                   | The full 15-layer (A–O) CDT pipeline + FastAPI/WebSocket server         |
| `static/index.html`         | The live operator dashboard (connects to `ws://host:8000/ws`)            |
| `weights/yolov8n_cdt.pt`    | A **real trained** YOLOv8n checkpoint (see "About the weights" below)    |
| `videos/demo_video.mp4`     | Synthetic crowd video, slow motion (tuned for low-FPS/CPU tracking)      |
| `videos/demo_video_fast_motion.mp4` | Same scenario, normal motion speed (for GPU-speed runs)          |
| `requirements.txt`          | Exact pip dependencies                                                   |
| `dataset/eth_ucy/`          | Empty — drop ETH/UCY `.txt` trajectory files here for real mobility priors (optional; pipeline runs fine without it) |

---

## 2. Quick start (any machine with Python 3.10+)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python main.py --source videos/demo.mp4
```

Then open **http://localhost:8000/** in a browser. You'll see the live
dashboard: agent counts, fused crowd count, risk score, zone grid, heatmap,
alerts feed, FPS/latency — all computed in real time from actual detections,
not mocked data.

Other source options:
```bash
python main.py --source 0                          # webcam
python main.py --source rtsp://<ip>:554/stream      # IP camera
python main.py --source videos/demo.mp4
```

REST endpoints (no browser needed):
```bash
curl http://localhost:8000/api/snapshot   # latest full JSON payload
curl http://localhost:8000/api/zones      # zone grid geometry
curl http://localhost:8000/api/fusion_log # last 20 fusion results
```

---

## 3. About the included weights — please read this

`weights/yolov8n_cdt.pt` was trained **inside the sandboxed assistant
environment**, which had **no GPU and no internet access** to dataset hosts
(`motchallenge.net`, `crcv.ucf.edu`, GitHub release-asset CDN for pretrained
COCO weights were all unreachable). So instead of fine-tuning the real
COCO-pretrained YOLOv8n on MOT17 + UCF-CC-50 + Mall like your report
describes, this checkpoint was trained from a random initialization on a
small **synthetic "person-blob" dataset** generated on the fly (120 train /
20 val images of colored humanoid blobs on dark backgrounds).

Result: **mAP50 = 0.981, mAP50-95 = 0.802, Precision = 0.982, Recall =
0.943** on its own validation set, and it reliably detects 19–22 of 22
walkers in the bundled demo videos. This proves the full pipeline — Layers
A through O — genuinely executes end-to-end with a real trained model, real
tracking, real fusion math, and a real live dashboard.

**It will NOT perform well on real CCTV footage of real people.** It has
never seen a real photograph — only synthetic blobs. To get the actual
system your report describes (mAP50 ≈ 0.216–0.22+ on real pedestrians,
70.8 FPS on a Tesla T4), you need to retrain it. Section 4 shows exactly how.

---

## 4. Getting the real, report-accurate model (GPU / Colab required)

Your original notebook (`crowd_digital_twin_complete.ipynb`) already does
this correctly. On a machine or Colab instance with internet + GPU access:

1. **Run Cell 1–2** to install dependencies and download MOT17, UCF-CC-50,
   Mall, and ETH/UCY into `/content/datasets/`.
2. **Run Cell 3** to convert all four datasets into unified YOLO format
   (this is what produces the 17,998-frame composite dataset described in
   Chapter 3 of your report).
3. **Run Cell 4** to fine-tune the *COCO-pretrained* `yolov8n.pt` (not
   `yolov8n.yaml` from scratch) for 50 epochs on a T4 GPU:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")   # pretrained, not random init
   model.train(data="composite_cdt.yaml", epochs=50, imgsz=640,
               batch=16, device=0, amp=False, patience=50)
   ```
4. Copy the resulting `runs/detect/train/weights/best.pt` into this
   package's `weights/yolov8n_cdt.pt`, replacing the synthetic-trained one.
5. Re-run `python main.py --source <your_video_or_rtsp>` — same code, now
   with the real model. You should see throughput close to the report's
   70.8 FPS on a T4-class GPU (vs. ~1.4 FPS observed on the sandbox's
   single CPU core).

Nothing else needs to change — `main.py`'s `Config.YOLO_MODEL` already
points at `weights/yolov8n_cdt.pt`, and the rest of the pipeline (fusion,
digital twin, risk estimator, dashboard) is dataset-agnostic.

### Loading ETH/UCY mobility priors
Drop the ETH/UCY trajectory `.txt` files (format: `frame  id  x  y`) into
`dataset/eth_ucy/` (any subfolder structure works — the loader globs
`**/*.txt`). If this folder is empty, the pipeline automatically falls back
to uniform zone calibration (1.0 weight) — it will not crash or block
startup either way (see Reliability, report §4.6).

---

## 5. Running on your own GPU machine (not Colab)

```bash
git clone <your repo, or just copy this folder>
cd CDT_Crowd_Digital_Twin
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Verify CUDA is visible to PyTorch (installed as an ultralytics dependency)
python -c "import torch; print(torch.cuda.is_available())"

python main.py --source videos/demo_video_fast_motion.mp4 --port 8000
```

If `torch.cuda.is_available()` is `False` on a machine with an NVIDIA GPU,
reinstall PyTorch with the correct CUDA build for your driver from
https://pytorch.org/get-started/locally/ before reinstalling `ultralytics`.

---

## 6. Performance notes (sandbox vs. report target)

| NFR                  | Report target | Report achieved (Tesla T4) | Observed in sandbox (1 CPU core) |
|-----------------------|---------------|------------------------------|-----------------------------------|
| Inference FPS          | ≥ 10 FPS      | 70.8 FPS                     | ~1.3–1.4 FPS                       |
| End-to-end latency     | < 2000 ms     | 14.1 ms                      | ~700–730 ms                        |
| Crowd count MAE        | < 5.0 persons/frame | 0.86 persons/frame    | N/A (no real ground truth in sandbox) |

The pipeline code itself is unchanged from what would run on a GPU — the
gap is purely the absence of a GPU and a single CPU core in the assistant's
execution sandbox. On any CUDA-capable machine (even a modest GTX 1650, per
your earlier 3DGS coursework), you should see large FPS gains immediately
because YOLOv8n inference and ByteTrack are the dominant per-frame costs and
both benefit heavily from GPU acceleration.

---

## 7. File-by-file map to the report's 15 layers (A–O)

| Layer | Report name | Class in `main.py` |
|-------|-------------|---------------------|
| A | Data Fusion Layer | `DataFusionLayer` |
| B | Edge AI: YOLOv8n Detection | `CDTPipeline._pipeline_loop` (model call) |
| C | ByteTrack Multi-Object Tracking | `sv.ByteTrack` (in `CDTPipeline.__init__`) |
| D | Confidence Estimator | `ConfidenceEstimator` |
| E | Zone Manager | `ZoneManager` |
| F | Crowd State (Agent Registry) | `Agent` dataclass, `DigitalTwinEngine.agents` |
| G | Adaptive Digital Twin Engine | `DigitalTwinEngine` |
| H | Density & Flow Estimator | `DensityFlowEstimator` |
| I | Trend Predictor | `TrendPredictor` |
| J | Simulation Trigger | inline in `CDTPipeline._pipeline_loop` (`sim_triggered`) |
| K | Short-Horizon Simulation | `DigitalTwinEngine.simulate_future` |
| L | Risk Estimator | `RiskEstimator` |
| M | Alert Engine | `AlertEngine` |
| N | Quantitative Evaluation | not wired into the live server; use the notebook's Cell 5/10 offline |
| O | Dashboard / Operator UI | `static/index.html` + WebSocket broadcast in `CDTPipeline` |

---

## 8. Troubleshooting

- **"Cannot open video source"** — check the path passed to `--source`, or
  that your webcam index / RTSP URL is correct and reachable.
- **Dashboard loads but stays "reconnecting"** — make sure `main.py` is
  running and bound to the same host/port the dashboard's `WS_URL` points
  at (edit the `ws-url` span / `WS_URL` JS constant in `index.html` if you
  changed `--port`).
- **Very low FPS on CPU** — expected; see Section 6. Use a GPU machine, or
  reduce `Config.YOLO_IMGSZ` (e.g. 320) and increase `Config.FRAME_SKIP` in
  `main.py` for a faster (lower-fidelity) CPU-only demo.
- **`weights/yolov8n_cdt.pt` missing** — `main.py` automatically falls back
  to `yolov8n.pt` (base COCO weights, 80 classes) if the fine-tuned file
  isn't found, per the Reliability section of the report. You'll still get
  person detection (COCO includes a "person" class), just not crowd-tuned.
