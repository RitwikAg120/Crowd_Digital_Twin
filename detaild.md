# Crowd Digital Twin (CDT) — In-Depth Technical Documentation

> **Project:** Crowd Digital Twin — Real-Time AI Crowd Monitoring & Simulation System
> **Team:** Team 28 | PES University | UE23CS320A Capstone Phase 2
> **Guide:** Dr. Richa Sharma
> **Members:** Rishikesh Suraj (CS481) · Rithvik Rajesh Matta (CS485) · Shreyas Sreenivas (CS565) · Ritwik Agrawal (CS908)
> **Repository:** `https://github.com/RitwikAg120/Crowd_Digital_Twin` (branch `dev`)

---

## Table of Contents

1. [What the System Does](#1-what-the-system-does)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Repository Layout](#4-repository-layout)
5. [Configuration Parameters](#5-configuration-parameters)
6. [Core Data Structures](#6-core-data-structures)
7. [End-to-End Implementation Walkthrough](#7-end-to-end-implementation-walkthrough)
8. [The Operator Dashboard (static/index.html)](#8-the-operator-dashboard)
9. [Web Server & REST/WebSocket API](#9-web-server--restwebsocket-api)
10. [Model Training Pipelines](#10-model-training-pipelines)
11. [Experience Buffer & Retrain Trigger](#11-experience-buffer--retrain-trigger)
12. [Running the System](#12-running-the-system)
13. [Non-Functional Requirements & Performance](#13-non-functional-requirements--performance)
14. [Known Limitations & Important Caveats](#14-known-limitations--important-caveats)

---

## 1. What the System Does

The **Crowd Digital Twin** is a real-time, CCTV/IoT-driven crowd-monitoring and
decision-support system built for **event safety and crowd-disaster prevention**.
It ingests a live video stream (webcam, RTSP IP camera, or a recorded file),
detects and tracks every person, fuses that vision data with simulated IoT
entry/exit gate sensors, and builds a live *digital twin* — a continuously
updated numerical model of the crowd scene.

From that twin, the system computes, in real time and from **real detections**
(not mocked numbers):

- **Tracked person count** and a *fused* count (vision + IoT).
- **Per-zone occupancy, density, and directional flow** over a 2×3 zone grid.
- A **scene-windowed trend** (GROWING / STABLE / DISPERSING) via linear regression.
- A **hybrid, confidence-weighted risk score** (LOW / MEDIUM / HIGH).
- **Zone-level alerts** and a scrolling incident feed for a human operator.
- **Short-horizon (≈20 s) what-if predictions** of where the crowd is heading.
- A live **heatmap overlay** of crowd density on the video feed.
- **FPS / latency** telemetry for non-functional-requirement (NFR) checks.

Everything is streamed over **WebSockets** to a browser dashboard styled as an
"Event Safety Command Centre". It is architected as an *edge-AI* system so that
a small on-premise machine can run detection close to the cameras.

The implementation intentionally follows a **15-layer (A–O) pipeline** described
in the team's capstone report, mapping each conceptual layer to a concrete class
or step in `main.py`.

---

## 2. High-Level Architecture

The conceptual end-to-end dataflow (pasted from the module docstring in
`main.py`):

```
CCTV/IoT/Mobility → VideoInputHandler → DataFusionLayer
  → Edge AI (YOLOv8n) → ByteTrack → ConfidenceEstimator
  → CrowdStateRepresentation + Heatmaps → ZoneManager
  → DensityFlowEstimator → DigitalTwinEngine (Agent-Based)
  → TrendPredictor → SimulationTrigger → ShortHorizonSimulation
  → RiskEstimator → AlertEngine → Dashboard → Human Operator
```

A more concrete runtime topology:

```
              main.py (CDTPipeline)
  Video source (webcam/RTSP/mp4)
      | frame  (VideoInputHandler background capture thread)
      v
  YOLOv8n detection  (classes=[0] person)
      | detections
      v
  ByteTrack MOT (sv.ByteTrack)
      | tracked boxes → det_list
      v
  DataFusionLayer (video + IoT + mobility prior)
      | fused_detections / fused_count
      v
  DigitalTwinEngine.update()  (live Agent model)
      | agents, zones, velocities
      v
  DensityFlowEstimator → TrendPredictor
      | density/flow · trend
      v
  RiskEstimator → AlertEngine
      | risk · alerts
      v
  ShortHorizonSimulation (simulate_future)
      | predicted zones
      v
  heatmap overlay → JSON payload
      |
  ExperienceBuffer  ◀── (offline learning)
      v
  WebSocket /ws broadcast (10 Hz) → Dashboard (static/index.html)
  REST /api/snapshot, /api/zones, /api/fusion_log
```

**Concurrency model:** the pipeline runs in a **dedicated daemon thread**
(`CDTPipeline._pipeline_loop`) so it never blocks the FastAPI event loop. Video
capture runs in **another daemon thread** (`VideoInputHandler._capture_loop`)
that continuously reads and caches the newest frame behind a lock, so the
pipeline never waits on I/O. Broadcasting to browsers is marshalled onto the
running asyncio loop via `asyncio.run_coroutine_threadsafe(...)`.

---

## 3. Tech Stack

| Area | Technology |
|------|-----------|
| Language | Python 3.10+ |
| Detection model | Ultralytics **YOLOv8n** (`yolov8n_cdt.pt`, fallback `yolov8n.pt`) |
| Multi-object tracking | **ByteTrack** via `supervision` (`sv.ByteTrack`) |
| Computer vision | OpenCV (`opencv-python-headless`) |
| Numerics | NumPy, SciPy (`gaussian_filter`, `linregress`) |
| Data / tables | Pandas (ETH/UCY mobility priors) |
| Web framework | FastAPI + Uvicorn |
| Real-time transport | WebSockets (dashboard), REST (JSON) |
| Static frontend | Vanilla HTML/JS/CSS (no build step), IBM Plex / Google Fonts |
| Cloud/network (notebook only) | `pyngrok` public tunnel from Colab |
| Persistence / learning | JSONL logs + sampled JPEG frames (experience buffer) |

**Key libraries (`requirements.txt`):** `ultralytics>=8.2.0`,
`supervision>=0.21.0`, `fastapi>=0.111.0`, `uvicorn[standard]>=0.30.1`,
`opencv-python-headless>=4.10.0`, `scipy>=1.13.0`, `numpy>=1.26.4`,
`pandas>=2.2.2`, `python-multipart>=0.0.9`, `pyngrok>=7.1.6`.

---

## 4. Repository Layout

```
Crowd_Digital_Twin/
├── main.py                          # Entire pipeline + FastAPI/WebSocket server
├── README.md                        # Quick start, weights notes, troubleshooting
├── detaild.md                       # This document
├── requirements.txt                 # Exact pip dependencies
├── static/
│   └── index.html                   # Live operator dashboard (single file)
├── weights/
│   └── yolov8n_cdt.pt               # Bundled synthetic-trained YOLOv8n checkpoint
├── videos/
│   ├── demo_video.mp4               # Slow-motion synthetic crowd demo (CPU-safe)
│   └── demo_video_fast_motion.mp4   # Normal-speed synthetic crowd demo (GPU)
├── dataset/
│   └── eth_ucy/                     # Optional mobility-prior trajectory data
├── experience/
│   ├── experience_buffer.py         # Adaptive-learning log + frame capture
│   └── __init__.py
├── retrain/
│   └── retrain_trigger.py           # Offline retraining orchestrator skeleton
├── training_scripts/
│   ├── README.md
│   ├── gen_synth.py                 # Synthetic person-blob dataset generator
│   ├── 01_train_synthetic_demo_model.py  # Trains the bundled checkpoint
│   └── gen_demo_video_slow.py       # Generates the slow-motion demo video
└── reference/
    ├── Capstone_Report_Team_28.docx            # The actual capstone report
    └── crowd_digital_twin_training_notebook.ipynb  # Real training pipeline
```

---

## 5. Configuration Parameters

All tuning constants live in the `Config` class at the top of `main.py`:

| Group | Constant | Value | Meaning |
|-------|----------|-------|---------|
| Model | `YOLO_MODEL` | `"weights/yolov8n_cdt.pt"` | Fine-tuned checkpoint (fallback: `yolov8n.pt`) |
| Model | `YOLO_CONF` | `0.35` | Detection confidence threshold |
| Model | `YOLO_IOU` | `0.50` | NMS IoU threshold |
| Model | `YOLO_IMGSZ` | `640` | Inference input size |
| Pipeline | `FRAME_SKIP` | `2` | Process every Nth captured frame |
| Pipeline | `FRAME_WIDTH/HEIGHT` | `1280×720` | Working resolution |
| Zones | `ZONE_ROWS/COLS` | `2×3` | Zone grid |
| Risk | `DENSITY_LOW/HIGH` | `0.15 / 0.40` | Persons per 10k px² thresholds |
| Risk | `VELOCITY_THR` | `25.0` | px/frame alarm velocity |
| Risk | `RISK_WINDOW` | `15` | Trend regression window |
| Sim | `PRED_HORIZON` | `25` | Simulation steps (~20 s) |
| Sim | `TRACKER_BUFFER` | `30` | ByteTrack lost-track buffer |
| Fusion | `W_VIDEO / W_IOT` | `0.70 / 0.30` | Stream fusion weights |
| Fusion | `DISCREP_THR` | `5` | Persons before flagging discrepancy |
| IoT sim | `IOT_NOISE_STD` | `2.0` | Sensor noise std-dev |
| IoT sim | `IOT_ENTRY_RATE/EXIT_RATE` | `0.10 / 0.08` | Entry/exit fraction per tick |
| Density | `HEATMAP_SIGMA` | `30.0` | Gaussian blur sigma |
| WebSocket | `WS_HZ` | `10` | Max broadcasts/sec |
| Data | `ETH_UCY_PATH` | `"dataset/eth_ucy"` | Optional mobility data |

---

## 6. Core Data Structures

**`Agent`** (dataclass) — the live digital-twin entity for each tracked person:
`id`, `x`, `y` (current centroid), `vx`, `vy` (velocity px/frame), `speed`,
`confidence` (fused belief), `zone` (assigned zone label), and a bounded
`history` (last ≤12 positions) used by predictors.

**`ZoneState`** (dataclass) — geometry for one zone: `name`, `x1/y1/x2/y2`,
`capacity` (default 50), plus computed properties `area`, `cx`, `cy`.

**`Config`** — static namespace of every tunable constant (see §5).

**Pipeline payload** (dict broadcast over WebSocket; also stored as
`latest_payload`) carries: `frame_idx`, `timestamp`, `n_agents`, `fused_count`,
`mean_speed`, `mean_confidence`, `risk_score`, `risk_label`, `crowd_trend`,
`trend_slope`, `trend_r2`, `sim_triggered`, `fps`, `latency_ms`, a nested
`fusion` object (video/IoT/fused counts, confidence, discrepancy, occlusion
flag, entry/exit deltas, conf_scale), a `zones` dict (per-zone
count/density/risk/risk_score/flow_x/flow_y/predicted/capacity), `alerts`, an
`agents` list, the `heatmap_b64` JPEG, and an `nfr` pass/fail object.

---

## 7. End-to-End Implementation Walkthrough

All logic is in **`main.py`**, a single ~1017-line module. A class named for
each conceptual layer wires together in `CDTPipeline`. This is the exact
per-frame flow.

### 7.1 Video Input Layer

**`VideoInputHandler`** owns a `cv2.VideoCapture`. `start()` opens the source
and launches a daemon `_capture_loop` thread that continuously reads frames
and caches the newest one behind a mutex (lock + `.copy()` on `read()`).
Recorded sources **auto-loop** (on `read()` failure it seeks back to frame 0),
giving a pseudo-live demo. It exposes `read()` and `resolution`.

### 7.2 Edge AI Detection (YOLOv8n)

`CDTPipeline._pipeline_loop` calls:

```python
results = self.model(frame, classes=[0], conf=Config.YOLO_CONF,
                     iou=Config.YOLO_IOU, imgsz=Config.YOLO_IMGSZ, verbose=False)
dets = sv.Detections.from_ultralytics(results[0])
```

Only **class 0 (person)** is detected. The model is either the bundled
`weights/yolov8n_cdt.pt` or, if missing, the COCO base `yolov8n.pt` (which
still detects persons). This is the **Layer B (Edge AI)** step.

### 7.3 Multi-Object Tracking (ByteTrack)

```python
self.tracker = sv.ByteTrack(track_activation_threshold=Config.YOLO_CONF,
                            lost_track_buffer=Config.TRACKER_BUFFER,
                            minimum_matching_threshold=0.8, frame_rate=25)
tracked = self.tracker.update_with_detections(dets)
```

ByteTrack bridges frames so the same person keeps a stable `tracker_id`. The
pipeline converts each tracked box to a dict `{id, cx, cy, confidence, zone}`
(the centroid midpoint, plus the zone it falls in via `ZoneManager.assign`).
This is **Layer C**.

### 7.4 Zone Manager

**`ZoneManager`** divides the frame into a **2×3 grid** of `ZoneState`s named
`Zone_A`…`Zone_F`. Capacity is scaled proportionally to each cell's area against
a nominal 150. It provides `assign(cx, cy)` (point-in-rect) and `count(agents)`
(per-zone tallies). This is **Layer E**.

### 7.5 Confidence Estimator

**`ConfidenceEstimator`** (Layer D) computes a **per-agent belief**:

```
conf = det_conf × age_weight × zone_calibration_weight × density_weight
```

- `age_weight` ramps from 0 → 1 over the first 5 frames (new tracks start untrusted).
- `zone_calibration_weight` comes from the ETH/UCY mobility prior (or defaults to 1.0).
- `density_weight` = `max(0.6, 1 − zone_density×0.4)` (dense zones → lower conf).

`set_zone_calibration(zone, w)` ingests the fused mobility priors; `drop(id)`
cleans house when a track disappears.

### 7.6 Data Fusion Layer (+ IoT Simulator)

**`IoTSimulator`** (Layer "Stream 2") simulates physical entry/exit gate
counters. `simulate_tick()` adds a Gaussian-noise entry count and a smaller noisy
exit count each frame and tracks cumulative `net_count`. In production you would
replace it with an MQTT subscriber calling `push_real(entry, exit)` (documented
inline).

**`DataFusionLayer.fuse(video_detections, frame_idx)`** (Layer A) implements the
report's weighted fusion:

```
C_f  = W_VIDEO×C_video + W_IOT×C_iot          (0.70 / 0.30)
κ    = 1 − |C_video − C_iot| / max(C_video, C_iot, 1)     # fusion confidence
```

Key logic: if **discrepancy > DISCREP_THR (5)**:
- IoT > video → assumed **occlusion** → *boost* detection confidence
  (`conf_scale = min(1, 1 + δ×0.02)`).
- video > IoT → possible *false positives* → *lower* confidence
  (`conf_scale = max(0.6, 1 − δ×0.02)`).

Each detection's confidence is re-scaled by `conf_scale × zone_mobility_weight`,
and the fused count is the weighted sum. The result dict is stored in a rolling
`_log` (deque, maxlen 300) exposed via `/api/fusion_log`.

`load_mobility_prior(eth_ucy_root, zone_mgr)` (Layer Stream 3 / mobility prior)
scans `dataset/eth_ucy/**/*.txt` for ETH/UCY trajectory files (`frame pid x y`),
computes a global mean pedestrian speed, and derives a per-zone confidence
weight. If the folder is empty or unreadable it **silently falls back to uniform
weight 1.0** — the pipeline still works (deliberate reliability design).

### 7.7 Digital Twin Engine / Agent Registry

**`DigitalTwinEngine`** (Layer G) maintains `self.agents: Dict[int, Agent]`, the
living model of the crowd. `update(detections, density_per_zone, fused_count)`:

1. Increments the frame counter.
2. For each detection: assigns a zone, recomputes a confidence via the
   `ConfidenceEstimator`, then either **updates an existing agent** — smoothing
   velocity with an **EMA `ALPHA = 0.4`**,
   `vx = ALPHA×(xn−a.x) + (1−ALPHA)×a.vx`, updating speed/zone/confidence and
   appending to bounded history — or **creates a new Agent**.
3. **Prunes stale tracks** not seen this frame (and tells the confidence
   estimator to drop them).
4. Appends a state snapshot (frame, n_agents, fused_count, mean_speed) to a
   rolling `state_log` (maxlen `RISK_WINDOW*4`).

This `Agent` dataclass + registry is the **Layer F (Crowd State / Agent
Registry)**.

### 7.8 Density & Flow Estimator

**`DensityFlowEstimator`** (Layer H):

- `density_map(agents, h, w)` — places a +1 at each agent centroid into a float
  grid, applies `gaussian_filter(sigma=30)`, and **normalizes to [0,1]**. This is
  the heatmap source.
- `zone_density(agents)` — per-zone count / area (persons per 10k px²).
- `zone_flow(agents)` — per-zone mean `(vx, vy)`.

### 7.9 Trend Predictor

**`TrendPredictor`** (Layer I) keeps rolling deques of `counts` and `speeds`
(maxlen `RISK_WINDOW`). `analyze()` needs ≥3 points, then runs
`scipy.stats.linregress` on the count series. Slope > `SLOPE_THR (0.15)` ⇒
**GROWING**; slope < −0.15 ⇒ **DISPERSING**; else **STABLE**; returns slope + R²
too.

### 7.10 Risk Estimator

**`RiskEstimator`** (Layer L) computes the report's hybrid risk score:

```
R = 0.40×density_norm + 0.30×speed_norm + 0.20×trend_score + 0.10×(1−confidence)
```

Each input is normalized (density & speed by 2× their threshold) and clipped to
[0,1]; trend maps GROWING=1.0 / STABLE=0.3 / DISPERSING=0.0. Classification:
`>0.66` ⇒ **HIGH**, `>0.33` ⇒ **MEDIUM**, else **LOW**. It is applied both
globally (scene) and per-zone (using that zone's density, mean speed, mean
confidence).

### 7.11 Alert Engine

**`AlertEngine`** (Layer M) converts each HIGH/MEDIUM zone risk into a human
message, e.g. `"CRITICAL: Zone_A — 18 agents, HIGH density..."` and `"WARNING:
Zone_B — ... approaching threshold."`. Only the first 5 alerts are sent.

### 7.12 Short-Horizon Simulation

The loop computes `sim_triggered = trend == "GROWING" or risk in (HIGH, MEDIUM)`.
If triggered, it calls `DigitalTwinEngine.simulate_future(PRED_HORIZON, h, w)`
(Layer K) — a lightweight **agent-based extrapolation**: for each live agent it
projects `x += vx`, `y += vy` for 25 steps with **boundary reflection**
(velocity flips on edges). The final state is re-zoned to produce
**`predicted` counts per zone** (what the crowd looks like ~20 s ahead). This is
the **Simulation Trigger (J) + Short-Horizon Simulation (K)**.

### 7.13 Heatmap Rendering

A normalized density map is scaled to 0–255 and `applyColorMap(JET)`; it is
alpha-blended (`0.55` frame / `0.45` heat) with the resized frame. Zone
rectangles are drawn colour-coded by risk (HIGH=red, MEDIUM=orange, LOW=green)
with a text label, and each tracked agent is marked with a dot. The composite is
JPEG-encoded (quality 72) and **base64-embedded** in the payload as
`heatmap_b64` so the browser can show it as a still image.

### 7.14 WebSocket Payload

The fully assembled `payload` dict (see §6) is stored in `latest_payload` (for
REST fallback) and broadcast at most `WS_HZ` (10 Hz) through the
`ConnectionManager`. FPS is a rolling mean (maxlen 30) of
`1/(frame_loop_duration)`; latency is the per-frame loop time in ms; both feed
the `nfr` pass/fail booleans (`fps >= 10`, `latency < 2000` ms).

### 7.15 Experience Buffer (Adaptive Learning)

After building the payload, the pipeline records an experience sample via
`ExperienceBuffer.record()`: it stores the frame index, timestamp, lightweight
detection/agent summaries, key metadata, and (every `frame_save_interval=30`
frames) a down-scaled JPEG frame to `experience/frames/`. Every 100 buffered
items, it spawns a background thread calling `flush()` to append a JSONL file to
`experience/logs/experience_<ts>.jsonl`. This is the mechanism for **offline
curation → retraining**, i.e. adaptive learning over the crowd's life.

---

## 8. The Operator Dashboard (static/index.html)

A **single-file** vanilla HTML/JS/CSS dashboard (no framework, no build). It
connects to `WS_URL = ws(s)://<host>/ws` (auto-derived from `location.host`) and
re-renders from each pushed payload. Layout:

- **Header / sub-header:** title, live badge, WS status pill, and a scenario
  selector (Mall, Concert surge, Temple gathering, Kumbh Mela, Evacuation) —
  cosmetic narrative presets for the demo.
- **Metrics row:** tracked persons (with delta + sparkline), scene risk score
  (pill + progress bar), crowd trend (slope · R²), and simulation status.
- **Left sidebar:** pipeline KV (model YOLOv8n, tracker ByteTrack, FPS, latency,
  mean conf, frame-skip, horizon, zones), NFR pass/fail checks, dataset tags,
  and fusion-weight readouts + live IoT entry/exit/occlusion.
- **Main content:** the live heatmap JPEG (`cam-feed`), overlaid risk labels, an
  FPS badge, and the 2×3 **zone grid** where each cell shows count, a fill bar
  vs. capacity, risk label, clickable details (density, Fruin LOS, flow,
  predicted), and predicted count.
- **Right sidebar:** incident alert feed (coloured, time-stamped), the fusion
  stream-comparison panel, a simulation panel (trend slope, predicted total,
  Fruin Level-of-Service estimate), and a scrolling ticker.
- **NFR widgets** colour FPS/latency pass–fail live.

The JS keeps rolling history buffers (agents, speed) for mini sparklines,
maintains `alertHistory` (last 40), and implements auto-reconnect with a 2 s
retry.

---

## 9. Web Server & REST/WebSocket API

FastAPI app `app = FastAPI(title="Crowd Digital Twin", version="2.0")` with open
CORS (`*`). A module-global `pipeline: CDTPipeline` holds the running instance;
`app_loop` caches the asyncio loop at startup for cross-thread broadcast.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ws` | WebSocket | Pushes the full JSON payload at up to 10 Hz; the browser also sends pings every 5 s |
| `/api/snapshot` | GET | Last full payload (heatmap stripped) as JSON; `{"status": "pipeline not started"}` if idle |
| `/api/zones` | GET | Zone geometry + capacities |
| `/api/fusion_log` | GET | Last 20 fusion results (debug) |
| `/` | GET | Serves `static/index.html` |
| `/static/*` | static | Mounted static assets |

Startup/shutdown lifecycle: `@app.on_event("startup")` calls `pipeline.start()`
(only if a pipeline was constructed); `shutdown` calls `pipeline.stop()`. The
`ConnectionManager` tracks active sockets and prunes dead ones on broadcast.

---

## 10. Model Training Pipelines

There are **two** training pipelines in the repo; it is critical to understand
the difference.

### 10.1 The Real (Report) Pipeline — `reference/` notebook

`reference/crowd_digital_twin_training_notebook.ipynb` (≈16 cells) is the
**real, report-accurate** training pipeline intended to run on **Google Colab
with GPU + internet**. Its cells:

1. **Install dependencies** (local Windows or Colab).
2. **Download datasets** — MOT17, UCF-CC-50, ETH/UCY, Mall.
3. **Convert all datasets to YOLO format** (with a MOT17 annotation fix).
4. **Auto-detect datasets, sanitize labels, fine-tune YOLOv8n** on the merged
   pedestrian data; includes a `torch.load` backend patch for Ultralytics
   compatibility.
5. **Evaluate on UCF-CC-50** (density benchmark).
6. **Prepare video source.**
7. **Upload project files** from Drive.
8. **Start the real-time server + ngrok public tunnel** (`pyngrok`).
9. **Monitor the pipeline** in Colab output.
10. **Collect & save evaluation metrics.**

This produces the report figures (mAP50 ≈ 0.216–0.22+ on real pedestrians,
~70.8 FPS on a Tesla T4, MAE 0.86 persons/frame). **It requires internet + GPU
and is NOT runnable in the offline sandbox.**

### 10.2 The Bundled Synthetic Pipeline — `training_scripts/`

Because the sandbox that packaged this repo had **no GPU and no internet**, the
bundled `weights/yolov8n_cdt.pt` was produced differently, purely to prove the
pipeline runs end-to-end with a genuinely-trained detector:

- **`gen_synth.py`** — draws 120 train / 20 val synthetic **"person-blob"**
  images (head circle + body ellipse on dark noisy backgrounds) with YOLO labels.
- **`01_train_synthetic_demo_model.py`** — builds `YOLO("yolov8n.yaml")` from
  architecture (no pretrained weights) and trains 30 epochs on that synthetic
  set on CPU (256 px, batch 8).
- **`gen_demo_video_slow.py`** — renders the 1400-frame slow-motion synthetic
  crowd video (22 walkers, cluster bias) used for low-FPS-safe ByteTrack demos.

Result (on its own synthetic val set): **mAP50 = 0.981, mAP50-95 = 0.802,
P = 0.982, R = 0.943**, and it reliably tracks 19–22 of the 22 walkers in the
demo videos.

> **⚠ Important (see README §3):** the synthetic model **will not perform well
> on real CCTV footage** — it has never seen a photograph of a person. To get the
> real report-accurate model, run the notebook's Cells 1–5 (see §10.1).

---

## 11. Experience Buffer & Retrain Trigger

- **`experience/experience_buffer.py`** — the `ExperienceBuffer` class: a
  bounded in-memory deque (default 500) of lightweight JSON records; optionally
  saves sampled/resized frames (every N frames) as JPEGs; `flush()` atomically
  drains the deque into timestamped JSONL files under `experience/logs/`;
  `inspect(n)` returns the last n records. All access is thread-safe (mutex).
- **`retrain/retrain_trigger.py`** — a lightweight orchestrator: counts samples
  across `experience/logs/*.jsonl`; if `--min-samples` (default 200) is met, it
  prints a recommendation and, with `--run`, invokes the training script
  `training_scripts/01_train_synthetic_demo_model.py`. A skeleton to be extended
  for CI/MLflow/model-registry integration.

Together these form the **closed feedback loop**: the live system logs
experiences, a threshold triggers offline (re)training, and the improved model
can be dropped back into `weights/`.

---

## 12. Running the System

```bash
# From Crowd_Digital_Twin/
python -m venv venv
venv\Scripts\activate            # (Windows)  /  source venv/bin/activate (Linux/macOS)
pip install -r requirements.txt

# Run the real-time server
python main.py --source videos/demo_video.mp4
# Other sources:
python main.py --source 0                        # webcam
python main.py --source rtsp://<ip>:554/stream   # IP camera
python main.py --source videos/demo_video_fast_motion.mp4
python main.py --source <file> --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000/** for the live dashboard. The console prints
the dashboard, WebSocket, and REST URLs up front.

> **Note:** `main.py`'s default `--source` is `videos/mot17_demo.mp4`, but the
> repo ships `demo_video.mp4` / `demo_video_fast_motion.mp4` — pass one of those
> explicitly (or your own file).

---

## 13. Non-Functional Requirements & Performance

| NFR | Report target | Report achieved (Tesla T4) | Sandbox observed (1 CPU core) |
|-----|---------------|------------------------------|-------------------------------|
| Inference FPS | ≥ 10 FPS | 70.8 FPS | ~1.3–1.4 FPS |
| End-to-end latency | < 2000 ms | 14.1 ms | ~700–730 ms |
| Crowd-count MAE | < 5.0 persons/frame | 0.86 persons/frame | N/A (no real GT) |

The **pipeline code is unchanged** between GPU and CPU; the gap is purely the
sandbox's lack of a GPU and a single core. On any CUDA-capable machine, YOLOv8n
inference and ByteTrack (the dominant costs) accelerate dramatically. The
dashboard's NFR badges (`fps >= 10`, `latency < 2000`) reflect this live.

---

## 14. Known Limitations & Important Caveats

- **Synthetic weights only (by design):** `weights/yolov8n_cdt.pt` is trained on
  synthetic blobs, not real pedestrians. Real CCTV accuracy requires retraining
  via the notebook (§10.1).
- **IoT stream is simulated** unless you replace `IoTSimulator` with a real MQTT
  consumer calling `push_real(...)`.
- **ETH/UCY prior is optional** — empty `dataset/eth_ucy/` gracefully degrades
  to uniform zone weights.
- **Frame skip = every 2nd frame** and demo videos are deliberately slow so
  ByteTrack remains coherent at low CPU FPS; on GPU you can lower `FRAME_SKIP`
  and/or raise motion speed.
- **Default `--source` file (`mot17_demo.mp4`) is not bundled** — override it
  with one of the shipped `videos/*.mp4`.
- **Dashboard geometry/labels** (zone capacities, scenario presets) are partly
  cosmetic narrative tied to the grid; the live numbers themselves come from the
  real pipeline.
- **Layer N (Quantitative Evaluation)** is not wired into the live server — it
  runs offline in the notebook (Cells 5/10).

---

*This document was generated by reading the repository end-to-end and reflects
the current state of the `dev` branch. For the academic rationale, weights
provenance, and NFR analysis, see `README.md` and the reference report
`reference/Capstone_Report_Team_28.docx`.*