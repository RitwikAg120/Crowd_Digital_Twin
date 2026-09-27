# Crowd Digital Twin (CDT)

Real-time crowd monitoring: YOLO26 person + head detection, ByteTrack
tracking, video/IoT count fusion, a per-agent digital twin with a 20 s
social-force forecast, zone risk scoring and a live WebSocket dashboard.

## 1. Layout

| File / folder              | What it is                                                              |
|----------------------------|--------------------------------------------------------------------------|
| `main.py`                  | The full 15-layer (A–O) CDT pipeline + FastAPI/WebSocket server          |
| `evaluate.py`              | Layer N — benchmarks and accuracy metrics; results go to `results/`      |
| `calibrate.py`             | Ground calibration: click ≥4 floor points, or give the camera's height/tilt |
| `calibration/`             | Calibration files; `auto_<video>.json` are the saved automatic estimates |
| `static/index.html`        | The live operator dashboard (connects to `/ws` on the same host)         |
| `static/twin.html`         | The 3D digital twin with the 20 s forecast, at `/twin` (connects to `/ws/twin`) |
| `weights/yolo26strained.pt`| Full-body (FBOX) detector — used for tracking. `yolo26ntrained.pt` is the smaller variant |
| `weights/yoloheadv26s.pt`  | Head (HBOX) detector. `yoloheadv26n.pt` is the smaller variant           |
| `yolo26s.pt`, `yolo26n.pt` | Base (untuned) YOLO26 weights; `yolo26s.pt` is the FBOX fallback         |
| `reference/`               | Training notebooks (`yolo26smodel`, `yolo26nmodel`, `yoloheadv1`) and the capstone report |
| `results.csv`              | Training log of the FBOX model (50 epochs, final mAP50 ≈ 0.857)          |
| `experience/`              | `ExperienceBuffer` — logs per-frame results (`logs/`) and sampled frames (`frames/`) |
| `retrain/retrain_trigger.py` | Reports when enough experience samples exist to retrain              |
| `videos/`                  | Demo footage                                                             |
| `dataset/eth_ucy/`         | Optional ETH/UCY trajectory files for mobility priors                    |
| `requirements.txt`         | Full pipeline (PyTorch + YOLO26)                                         |
| `requirements-twin.txt`    | Twin-only mode — no PyTorch                                              |

## 2. Quick start (Python 3.10+)

There are two ways to run the project. Detection and training belong on a
GPU machine (the college GB10); the twin-only mode runs anywhere.

| Mode | What runs | Disk | RAM |
|------|-----------|------|-----|
| Full pipeline (`venv`) | video → YOLO26 body + head → ByteTrack → twin → dashboard | 1.4 GB env + 0.13 GB weights | ~450 MB idle, ~800 MB peak on CPU |
| Twin-only (`venv-twin`) | recorded tracks → fusion, twin, risk, 20 s forecast → dashboard | 0.36 GB env | ~150 MB |

Measured on the development laptop (CPU): the full pipeline runs at ~2–3 FPS
there; the twin alone handles ~200 people at ~11 ms per frame, and the 3D
twin's 20 s forecast for them takes ~0.3 s once a second.

**Full pipeline**
```bash
python -m venv venv
venv\Scripts\activate             # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt

python main.py --source videos/demo.mp4
```
or venv\Scripts\python main.py --source videos\demo.mp4

**Twin-only mode** (no PyTorch, no GPU)
```bash
python -m venv venv-twin
venv-twin\Scripts\activate
pip install -r requirements-twin.txt

python main.py --replay experience/logs                    # latest recorded run
python main.py --replay experience/logs/experience_20260830T033805.jsonl   # the run this file belongs to
python main.py --replay <MOT17>/train/MOT17-09-FRCNN       # MOT ground truth or tracker output
python main.py --replay tracks.txt --size 1920x1080 --fps 30   # any MOT-format CSV
```
Replay plays the tracks back in real time and loops, and drives the 3D twin
too. Experience logs from
before 27 Sep 2026 don't store the frame size, so it is inferred (override
with `--size`) and their walking speeds are approximate. Newer logs store it.
A GPU machine can run the detectors, keep the experience logs (or write MOT
CSV tracks), and the twin can then be replayed and developed on a laptop.

Open **http://localhost:8000/** for the dashboard and **http://localhost:8000/twin**
for the 3D digital twin with the 20 s forecast. The 3D twin loads Three.js from
the jsDelivr CDN, so the browser needs internet access.

Other sources and options:
```bash
python main.py --source 0                          # webcam
python main.py --source rtsp://<ip>:554/stream     # IP camera
python main.py --calibration calibration/cam1.json # measured ground calibration (see §4)
python main.py --hfov 80                           # camera's field of view, for the automatic estimate
python main.py --device cpu                        # default: auto (CUDA when available)
```

REST endpoints:
```bash
curl http://localhost:8000/api/snapshot   # latest payload (without the image)
curl http://localhost:8000/api/zones      # zone grid geometry and area in m²
curl http://localhost:8000/api/fusion_log # last 20 fusion results
curl http://localhost:8000/api/twin       # 3D twin: scene (floor, walls, zones, camera), crowd state, 20 s forecast
```

## 3. Models

The full-body (FBOX) detector feeds ByteTrack and the digital twin; the head
(HBOX) detector is drawn on the video and sent to the dashboard as
`head_bounding_boxes` (set `HBOX_EVERY` in `Config` to run it less often).
`evaluate.py ucf` also scores head and body+head counting via
`fuse_fbox_hbox()`.

Inference runs on CUDA in FP16 when PyTorch sees a GPU, otherwise on CPU.
Model paths, confidence thresholds, input sizes and `MAX_DET` (1000, so dense
crowds are not capped at Ultralytics' default of 300) live in `Config` at the
top of `main.py`. To retrain, use the notebooks in `reference/` (MOT17 +
CrowdHuman) and copy the resulting `best.pt` into `weights/`.

## 4. How the live pipeline works

- **Real-time input.** Video files play at their own frame rate and loop;
  every `FRAME_SKIP`-th frame (default 2nd) is processed. The camera is
  assumed to be fixed (CCTV): on handheld or panning footage, camera motion
  shows up as crowd speed.
- **Ground calibration (perspective).** Each person stands where their feet
  are (the bottom of the box); `GroundPlane` maps that pixel to metres on the
  floor with a homography, so zone areas, densities (persons/m²) and speeds
  (m/s) account for perspective — in an oblique view a zone at the top of the
  frame can cover 10× more floor than one at the bottom. The calibration
  comes from, in order:
  1. `--calibration file.json` — made with `calibrate.py`: click ≥4 floor
     points whose real positions you know (tiles, markings, a measured
     rectangle), or give the camera's height, tilt and field of view.
  2. An automatic estimate from people's box heights (`PedestrianCalibrator`,
     single-view metrology): a person's pixel height grows linearly with how
     far below the horizon their feet are, which gives the camera's height and
     tilt. It runs on the first ~20–300 frames, needs ~300 unobstructed
     full-body boxes, assumes a `--hfov` of 65° and people 1.7 m tall, and is
     saved to `calibration/auto_<video>.json` for reuse with `--calibration`.
     A wrong field of view mostly scales the depth direction (a 50° vs 80°
     guess changes far-zone areas by up to ~2×).
  3. A flat `--scene-width-m × --scene-height-m` scale (40 × 22.5 m) when
     neither is available (e.g. overhead drone views or very sparse scenes).

  The floor's far edge is learned from where people's feet have been seen:
  the image above it (walls, sky) is not counted as floor. It only ever moves
  up. The dashboard's **Ground** row shows which calibration is active.
- **Zones and risk.** Zone capacity is the head-count at 0.8 p/m² (Fruin LOS
  F), where the density risk term saturates; the speed term saturates at
  3 m/s (running) and uses each zone's median speed.
- **Fusion (A).** `C_f = 0.70·C_video + 0.30·C_iot`,
  `κ = 1 − |C_video − C_iot| / max(C_video, C_iot, 1)`. Without real gate
  sensors, `IoTSimulator` follows the smoothed video count plus people the
  camera misses, with a bounded (mean-reverting) miscount. Call
  `pipeline.fusion.iot.push_real(entry, exit)` from a sensor callback to switch
  to live counts. When the gates see more than the camera, detection
  confidence is boosted (up to ×1.2).
- **Lost tracks.** Agents that ByteTrack loses stay in the twin (not counted)
  for `TRACKER_BUFFER` = 30 frames and are purged after that.
- **Simulation trigger (J).** `SimulationTrigger` flags the forecast (with its
  reasons) when the crowd is GROWING, the scene risk is MEDIUM/HIGH, or
  κ < 0.70; the dashboard shows the flag and links to the 3D twin.
- **3D digital twin with the 20 s forecast (K), at `/twin`.** Separate from the
  dashboard: `TwinService` reads the pipeline's crowd state in metres and, while
  a viewer is open, runs a social force model (Helbing & Molnár, 1995) every
  second — 25 steps × 0.8 s = 20 s ahead, bouncing off the edges of the
  walkable floor. The viewer (`static/twin.html`, Three.js) builds its
  environment from the calibration: the walkable floor with a 2 m grid, the
  zones, **walls** where the floor ends (learned from people's feet), low
  markers at the **edges of the camera's view**, and the camera itself at its
  estimated height and tilt. People are low-poly instanced proxies (hexagonal
  prism + icosahedron, heading arrow) coloured by speed or zone risk. Scrub or
  play the 0–20 s timeline to see translucent forecast proxies, their paths,
  zones coloured by predicted density, and when a zone reaches capacity. Views:
  from the real camera (CCTV), 3/4 and top-down.
- **Heatmap.** A density map with a 1 m Gaussian per person on the ground (so
  people far away get smaller blobs) is blended under the boxes. Set
  `SEND_FRAME = False` to send the heatmap without any video pixels.
- **Telemetry.** `fps` is processed frames per second; `latency_ms` is
  end-to-end, from frame capture to finished payload; `timing_ms.detect` is
  the detector share.
- **Experience logs.** The live pipeline logs every processed frame to
  `experience/` for retraining (~0.35 GB/hour at CPU speed, ~1.6 GB/hour at
  12.5 FPS on a dense crowd). The folder is capped at `EXPERIENCE_MAX_GB`
  (2 GB): the oldest logs and frames are deleted first. Replay mode does not
  log.

### ETH/UCY mobility priors
Drop ETH/UCY trajectory `.txt` files (4-column `frame id x y`, tab or space
separated, or the 8-column ETH `obsmat` format) into `dataset/eth_ucy/` (the
loader globs `**/*.txt`). Without them the pipeline uses uniform zone
calibration; the dashboard shows which one is active.

## 5. Evaluation (Layer N)

```bash
python evaluate.py bench --source videos/demo.mp4 --frames 200   # end-to-end FPS / latency
python evaluate.py mot17 --seq <MOT17>/train/MOT17-09-FRCNN      # count MAE/RMSE/r + MOTA
python evaluate.py ucf --root <UCF_CC_50 folder>                 # count error, body/head/both
python evaluate.py ade --root dataset/eth_ucy                    # ADE/FDE: linear vs social force
```

Each run prints its metrics and saves them as JSON in `results/`. `bench`
measures the whole pipeline (decode → detection → tracking → twin → overlay →
JSON), not just the detector; `mot17` processes every frame and scores
tracking with CLEAR-MOT (MOTA, MOTP, ID switches). The datasets are not in the
repo — download MOT17, UCF-CC-50 and ETH/UCY first.

## 6. Layer map (report layers A–O)

| Layer | Report name | Where in `main.py` |
|-------|-------------|---------------------|
| A | Data Fusion Layer | `DataFusionLayer`, `IoTSimulator` (called from `TwinPipeline.process_tracks`) |
| B | Edge AI Detection | `CDTPipeline.detect` (FBOX + HBOX); `ReplayPipeline` replays recorded tracks instead |
| C | ByteTrack Multi-Object Tracking | `sv.ByteTrack` (in `CDTPipeline.configure`) |
| D | Confidence Estimator | `ConfidenceEstimator` |
| E | Zone Manager | `ZoneManager`, with ground areas from `GroundPlane` (+ `PedestrianCalibrator`) |
| F | Crowd State (Agent Registry) | `Agent` dataclass, `DigitalTwinEngine.agents` |
| G | Adaptive Digital Twin Engine | `DigitalTwinEngine` |
| H | Density & Flow Estimator | `DensityFlowEstimator` |
| I | Trend Predictor | `TrendPredictor` |
| J | Simulation Trigger | `SimulationTrigger` |
| K | Short-Horizon Simulation | `TwinService.forecast` + `SocialForceModel`, shown in the 3D twin (`/twin`) |
| L | Risk Estimator | `RiskEstimator` |
| M | Alert Engine | `AlertEngine` |
| N | Quantitative Evaluation | `evaluate.py` |
| O | Dashboard / Operator UI | `static/index.html` (2D dashboard, `/ws`) and `static/twin.html` (3D twin, `/ws/twin`) |

## 7. Troubleshooting

- **"Cannot open video source"** — check the `--source` path, webcam index
  or RTSP URL.
- **`HBOX weights not found`** — `weights/yoloheadv26s.pt` is required; the
  FBOX model falls back to `yolo26s.pt` but the head model has no fallback.
- **Dashboard shows `cpu` / FPS check fails** — PyTorch is the CPU-only build.
  Check `python -c "import torch; print(torch.cuda.is_available())"` and
  install the CUDA build from https://pytorch.org/get-started/locally/.
- **Still too slow** — two detectors at 960/1280 px are heavy. Lower
  `YOLO_IMGSZ` / `HBOX_IMGSZ`, raise `HBOX_EVERY`, or switch to the `n` weights.
- **`ByteTrack` import error** — supervision 0.31 removed it; keep
  `supervision<0.31` (pinned in `requirements.txt`).
- **Ground stays "flat"** — the automatic estimate needs unobstructed
  full-body boxes (people not cut off by the frame) and a camera that looks
  at people from the side; it can't work on overhead views or when the
  detector only finds a handful of people. Use `calibrate.py` instead.
- **Zone areas or speeds look off** — check the estimate in
  `calibration/auto_<video>.json`. If the camera's field of view differs a lot
  from 65°, pass `--hfov`; if you can measure a few floor points, a
  `calibrate.py` point calibration is more accurate than the estimate.
