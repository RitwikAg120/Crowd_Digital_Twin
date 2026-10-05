# Crowd Digital Twin (CDT)

Real-time crowd monitoring: YOLO26 person + head detection (with a dense-crowd
mode for crowds where only heads are visible), ByteTrack tracking, video/IoT
count fusion, a per-agent digital twin with a self-calibrating 20 s forecast,
zone risk scoring and a live WebSocket dashboard.

## 1. Layout

| File / folder              | What it is                                                              |
|----------------------------|--------------------------------------------------------------------------|
| `main.py`                  | The full 15-layer (A–O) CDT pipeline + FastAPI/WebSocket server          |
| `forecast.py`              | Layer K — motion filter, what the twin learns about the scene, the 20 s forecast and its self-check |
| `dense.py`                 | Dense-crowd head points (P2PNet); used when `weights/p2pnet_crowd_jhu.pth` (or `_crowd.pth`) exists |
| `iot.py`                   | Gate counters (Stream 2) over HTTP (`POST /api/iot`) or MQTT              |
| `evaluate.py`              | Layer N — benchmarks and accuracy metrics; results go to `results/`      |
| `train_dense.py`           | Trains the dense-crowd point model (run on the GB10)                     |
| `export.py`                | TensorRT / ONNX export for Jetson or the GB10 (run on the target)        |
| `scripts/`                 | GB10: `gb10_setup.sh`, `fetch_datasets.py`, `run_gb10.sh` (every measurement and training run) |
| `tools/`                   | `annotate_points.py` (label heads for training), `iot_gate_sim.py` (play a gate counter) |
| `tests/`                   | Checks: `python tests/test_forecast.py` (twin-only env), `test_dense.py`, `test_api.py` (full env) |
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
| `dataset/`                 | GB10 only: `MOT17/`, `CrowdHuman/`, `JHU-Crowd/` (not in git), `kumbh_points/` for your own labelled frames |
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
for the 3D digital twin with the 20 s forecast. Three.js is served from
`static/vendor/three/`, so everything runs without internet access.

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
curl http://localhost:8000/api/sources    # videos the dashboard can switch to
curl -X POST -H "Content-Type: application/json" -d '{"name":"demo.mp4"}' http://localhost:8000/api/source
curl http://localhost:8000/api/twin       # 3D twin: scene (floor, walls, zones, camera), crowd state, 20 s forecast
```

## 3. Models

The full-body (FBOX) detector feeds ByteTrack and the digital twin. Heads
from the head (HBOX) detector that have no body box around them are people
the body detector missed: each becomes a whole-person box (head on top, feet
from the ground geometry — exact with a camera calibration, 7 head-heights
otherwise) and is tracked with the bodies.

**Dense mode.** In a dense crowd (a ghat at the Kumbh) bodies are hidden and
heads are 3–10 px, so both detectors miss most people. When many heads have
no body (at least `DENSE_MIN_HEADS` = 40 of them, and ≥ `DENSE_HEAD_RATIO` = 2×
the bodies), the pipeline switches to dense mode by itself: the head detector
runs at `HBOX_IMGSZ_DENSE` (2560 px) and, if a point model is present, the
P2PNet point model (`dense.py`) finds one point per head — the model built for
dense crowds. The frame's contrast is equalised first (`DENSE_ENHANCE`) so heads
in fog and at dusk stand out, and each head point is sized for its image row
from the head detector's boxes (`dense.HeadScale`). On `kumbhvideo4.mp4` the twin
went from 3–23 tracked people per frame (bodies only) to several hundred with the
point model. The model is `weights/p2pnet_crowd_jhu.pth` if present (CrowdHuman +
JHU-Crowd++, the better one for dense/foggy/high-view crowds), else
`weights/p2pnet_crowd.pth` (CrowdHuman only); train either on the GB10
(`train_dense.py`, §5). The dashboard shows **Dense mode** and how people were
**found** (body / head / point).

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
     In dense crowds, where bodies are hidden, head boxes do the same job
     (heads are `HEAD_SIZE_M` = 0.25 m and sit on a plane 1.45 m above the
     floor); they are used first when they outnumber bodies 3 to 1.
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
  camera misses, with a bounded (mean-reverting) miscount. Real gate counters
  report over HTTP (`POST /api/iot {"entry": 3, "exit": 1}`, optional
  `--iot-token`) or MQTT (`--mqtt broker:1883 --mqtt-topic cdt/gates/#`, needs
  `paho-mqtt`); the first report switches the fusion to live gates.
  `tools/iot_gate_sim.py` plays a gate counter over either path. When the
  gates see more than the camera, detection confidence is boosted (up to ×1.2).
- **Motion state.** Each person's feet go through a Kalman filter on the
  ground (`forecast.MotionFilter`) whose measurement noise comes from the
  perspective: a far-away pixel spans metres of floor, so a far person's box
  jitter does not read as walking. Speeds are the filtered velocity; each
  person also carries its uncertainty and a walking/standing flag.
- **Lost tracks.** Agents that ByteTrack loses stay in the twin (not counted)
  for `TRACKER_BUFFER` = 30 frames and are purged after that.
- **Simulation trigger (J).** `SimulationTrigger` flags the forecast (with its
  reasons) when the crowd is GROWING, the scene risk is MEDIUM/HIGH, or
  κ < 0.70; the dashboard shows the flag and links to the 3D twin.
- **3D digital twin with the 20 s forecast (K), at `/twin`.** Separate from the
  dashboard: `TwinService` reads the pipeline's crowd state in metres and, while
  a viewer is open, forecasts 25 steps × 0.8 s = 20 s ahead every second with
  `forecast.CrowdForecaster`, an anticipatory social force model:
  walkers keep their heading and speed but turn towards the lanes the twin
  has learned (a flow field of where people walk); standing people stay put;
  people avoid only those they are about to meet (time-to-collision law,
  Karamouzas et al. 2014), so a dense standing crowd does not blow apart;
  walking slows with local density (Weidmann's fundamental diagram); people
  leave through the edges of the camera's view and slide along walls; and
  newcomers arrive where and as often as people have been seen entering.
  The twin also checks its own forecasts: `ForecastSkill` scores each zone
  count forecast against what then happened and learns, per horizon, how far
  to trust the predicted change over "no change" — the 3D twin shows that
  running skill. The viewer (`static/twin.html`, Three.js) builds its
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
  the detector share. `GET /api/health` reports whether video is arriving,
  uptime, FPS and stream reconnects, for long unattended runs.
- **Live streams.** An RTSP camera that stops delivering frames for ~2 s is
  reopened automatically, backing off up to 30 s between tries.
- **Experience logs.** The live pipeline logs every processed frame to
  `experience/` for retraining (~0.35 GB/hour at CPU speed, ~1.6 GB/hour at
  12.5 FPS on a dense crowd). The folder is capped at `EXPERIENCE_MAX_GB`
  (2 GB): the oldest logs and frames are deleted first. Replay mode does not
  log.

## 5. Evaluation (Layer N)

```bash
python evaluate.py bench --source videos/demo.mp4 --frames 200   # end-to-end FPS / latency
python evaluate.py tracks --source videos/demo.mp4 --out dataset/tracks/demo   # one pass → MOT tracks
python evaluate.py forecast --tracks dataset/tracks/demo          # 20 s forecast vs what happened
python evaluate.py mot17 --seq <MOT17>/train/MOT17-09-FRCNN --start-frac 0.85  # count MAE/RMSE/r + MOTA on held-out frames
python evaluate.py dense --data crowdhuman:dataset/CrowdHuman --max 500 --enhance   # crowd-count error
```

`mot17 --start-frac 0.85` scores only the last 15 % of each sequence — the
frames the body detector never trained on (its notebook held them out), so the
numbers aren't inflated. `dense` reports count error for bodies, bodies + heads
(normal and dense-mode resolution) and the point model, on CrowdHuman val
(`crowdhuman:`), JHU-Crowd++ val (`jhu:`) or your own labelled frames
(`points:`); `--enhance` applies the same contrast step the live pipeline uses.

Each run prints its metrics and saves them as JSON in `results/`. `bench`
measures the whole pipeline (decode → detection → tracking → twin → overlay →
JSON), not just the detector; `mot17` processes every frame and scores
tracking with CLEAR-MOT (MOTA, MOTP, ID switches). `forecast` replays tracks in
time order like the live twin and, every second, forecasts with each model —
persistence (nobody moves), constant velocity, the old twin (1 s slope
velocity + Helbing social force bouncing off the view's edges) and the new
twin, with ablations — then scores people's positions (ADE/FDE) and zone
counts at 2.4–20 s against what actually happened.

**Datasets and training run on the GB10**, not the laptop. The project uses only
**MOT17**, **CrowdHuman** and **JHU-Crowd++**:
```bash
bash scripts/gb10_setup.sh                 # venv + ARM64 CUDA PyTorch + requirements
python scripts/fetch_datasets.py mot17     # MOT17 02/04/09 only, via HTTP range requests
python scripts/fetch_datasets.py crowdhuman jhu   # checks these are in place (licence-gated downloads)
bash scripts/run_gb10.sh [measure|mot17|train|jhu|all]   # benchmarks, evaluation, dense-model training
```
CrowdHuman (crowdhuman.org) and JHU-Crowd++ (crowd-counting.com) need their
licences accepted, so `fetch_datasets.py` only checks they are unpacked under
`dataset/`. `run_gb10.sh train` trains the point model on CrowdHuman; `jhu`
fine-tunes it on CrowdHuman + JHU-Crowd++ (`weights/p2pnet_crowd_jhu.pth`). To
adapt it to your own footage, label 30–50 frames with `tools/annotate_points.py`
(it pre-fills heads from the detector) into `dataset/kumbh_points/{train,test}`;
`run_gb10.sh` includes them.

## 6. Layer map (report layers A–O)

| Layer | Report name | Where in `main.py` |
|-------|-------------|---------------------|
| A | Data Fusion Layer | `DataFusionLayer`, `IoTSimulator` (called from `TwinPipeline.process_tracks`) |
| B | Edge AI Detection | `CDTPipeline.detect` (FBOX + HBOX), dense mode + `dense.PointCounter`; `ReplayPipeline` replays recorded tracks instead |
| C | ByteTrack Multi-Object Tracking | `sv.ByteTrack` (in `CDTPipeline.configure`) |
| D | Confidence Estimator | `ConfidenceEstimator` |
| E | Zone Manager | `ZoneManager`, with ground areas from `GroundPlane` (+ `PedestrianCalibrator`) |
| F | Crowd State (Agent Registry) | `Agent` dataclass, `DigitalTwinEngine.agents` |
| G | Adaptive Digital Twin Engine | `DigitalTwinEngine` |
| H | Density & Flow Estimator | `DensityFlowEstimator` |
| I | Trend Predictor | `TrendPredictor` |
| J | Simulation Trigger | `SimulationTrigger` |
| K | Short-Horizon Simulation | `TwinService.forecast` + `forecast.py` (`CrowdForecaster`, `SceneMemory`, `ForecastSkill`), shown in the 3D twin (`/twin`); `SocialForceModel` is kept as the evaluation baseline |
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
  Dense mode's 2560 px head pass takes ~1 s per frame on a laptop CPU; on a
  GPU it is fast — or lower `HBOX_IMGSZ_DENSE`, or set `DENSE_AUTO = False`.
  On Jetson / GB10, `python export.py` builds TensorRT engines; run with
  `--body weights/yolo26strained.engine --head weights/yoloheadv26s.engine`.
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
