# Crowd Digital Twin (CDT) — In-Depth Technical Documentation

> **Project:** Crowd Digital Twin — Real-Time AI Crowd Monitoring & Simulation System
> **Team:** Team 28 | PES University | UE23CS320A Capstone
> **Guide:** Dr. Richa Sharma
> **Members:** Rishikesh Suraj (CS481) · Rithvik Rajesh Matta (CS485) · Shreyas Sreenivas (CS565) · Ritwik Agrawal (CS908)
> **Repository:** `https://github.com/RitwikAg120/Crowd_Digital_Twin` (branch `DTv2`)

This document describes the **current** system: YOLO26 body + head detection with
a dense-crowd point model, a perspective-calibrated digital twin, a
self-calibrating 20 s forecast, and a 3D twin with a panic what-if. It is kept in
sync with the code; `README.md` is the quick-start, this is the deep dive.

---

## Table of Contents

1. [What the System Does](#1-what-the-system-does)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Repository Layout](#4-repository-layout)
5. [Configuration Parameters](#5-configuration-parameters)
6. [Core Data Structures](#6-core-data-structures)
7. [End-to-End Implementation Walkthrough](#7-end-to-end-implementation-walkthrough)
8. [Detection, Dense Mode & the Point Model](#8-detection-dense-mode--the-point-model)
9. [Perspective Ground Calibration](#9-perspective-ground-calibration)
10. [The Forecast (forecast.py)](#10-the-forecast-forecastpy)
11. [The Dashboard & 3D Twin](#11-the-dashboard--3d-twin)
12. [Web Server & REST/WebSocket API](#12-web-server--restwebsocket-api)
13. [Datasets, Training & the GB10](#13-datasets-training--the-gb10)
14. [Evaluation (Layer N)](#14-evaluation-layer-n)
15. [Non-Functional Requirements & Performance](#15-non-functional-requirements--performance)
16. [Known Limitations & Caveats](#16-known-limitations--caveats)

---

## 1. What the System Does

The **Crowd Digital Twin** is a real-time, CCTV/IoT-driven crowd-monitoring and
decision-support system for **event safety and crowd-disaster prevention**. It
ingests a live video stream (webcam, RTSP IP camera, or a recorded file), detects
and tracks every person, fuses that with simulated (or real) IoT entry/exit gate
counts, and maintains a live *digital twin* — a continuously updated numerical
model of the crowd on the real ground plane, in metres.

From that twin, from **real detections** (not mocked numbers):

- **Tracked person count** and a *fused* count (vision + IoT).
- **Per-zone occupancy, density (persons/m²) and directional flow** over a 2×3 grid.
- A **trend** (GROWING / STABLE / DISPERSING) by linear regression on the count.
- A **confidence-weighted risk score** (LOW / MEDIUM / HIGH), global and per zone.
- **Zone alerts** and a scrolling incident feed for an operator.
- A **self-calibrating ~20 s forecast** of where the crowd is heading, in a 3D twin.
- A **panic what-if**: an escape-panic simulation and where a crush could form.
- A live **density heatmap** overlay, and **FPS / latency** telemetry.

Everything streams over **WebSockets** to a browser dashboard. It is an *edge-AI*
system: a small on-premise machine runs detection close to the cameras, and the
twin can be replayed and developed on a laptop with no GPU.

The implementation follows a **15-layer (A–O) pipeline** from the team's report,
mapping each layer to a concrete class in `main.py` (see the layer map in §7).

---

## 2. High-Level Architecture

Conceptual dataflow (from the `main.py` module docstring):

```
CCTV/IoT → VideoInputHandler → DataFusionLayer
  → Edge AI (YOLO26s body + head detectors) → ByteTrack → ConfidenceEstimator
  → CrowdStateRepresentation + Heatmaps → ZoneManager
  → DensityFlowEstimator → DigitalTwinEngine (Agent-Based)
  → TrendPredictor → SimulationTrigger → ShortHorizonSimulation (forecast)
  → RiskEstimator → AlertEngine → Dashboard → Human Operator
```

**Concurrency.** The detection/twin pipeline runs in a dedicated daemon thread
(`CDTPipeline._pipeline_loop`) so it never blocks the FastAPI event loop. Video
capture runs in another daemon thread (`VideoInputHandler._capture_loop`) that
caches the newest frame behind a lock, so the pipeline never waits on I/O. The 3D
twin's forecast runs in a third thread (`TwinService`) while a viewer is open.
Browser broadcasts are marshalled onto the asyncio loop with
`asyncio.run_coroutine_threadsafe`.

**Two run modes.** The full pipeline (`venv`, PyTorch + YOLO26) does detection;
twin-only mode (`venv-twin`, no PyTorch) replays recorded MOT tracks through the
same twin, fusion, risk and forecast — so the twin is developed on a laptop while
detection and training happen on the GB10.

---

## 3. Tech Stack

| Area | Technology |
|------|-----------|
| Language | Python 3.10+ |
| Body + head detection | Ultralytics **YOLO26s** (`yolo26strained.pt`, `yoloheadv26s.pt`; base fallback `yolo26s.pt`) |
| Dense-crowd counting | **P2PNet** head-point model (`dense.py`), VGG16-BN backbone |
| Multi-object tracking | **ByteTrack** via `supervision` (`sv.ByteTrack`) |
| Computer vision | OpenCV (`opencv-python-headless`) |
| Numerics | NumPy, SciPy (`gaussian_filter`, `least_squares`, `cKDTree`, `linregress`, `linear_sum_assignment`) |
| Web framework | FastAPI + Uvicorn |
| Real-time transport | WebSockets (dashboard + 3D twin), REST (JSON) |
| Frontend | Vanilla HTML/JS/CSS (no build step); Three.js served locally from `static/vendor/three/` |
| IoT (optional) | MQTT via `paho-mqtt` (`iot.py`) |
| Persistence / learning | JSONL logs + sampled JPEG frames (experience buffer) |

Ultralytics runs with `YOLO_OFFLINE=1` (no update checks / telemetry), so the full
pipeline works without internet.

---

## 4. Repository Layout

```
Crowd_Digital_Twin/
├── main.py            # Pipeline, ground plane, dense mode, twin, 3D twin service, what-if, API
├── forecast.py        # Motion filter, scene memory, forecaster, self-calibration, panic model
├── dense.py           # P2PNet head-point model, PointCounter, HeadScale, contrast enhance
├── stabilize.py       # Camera-motion compensation (CameraMotion)
├── iot.py             # Gate counters over HTTP / MQTT
├── export.py          # TensorRT / ONNX export for Jetson / GB10
├── evaluate.py        # Layer N: bench | tracks | mot17 | forecast | dense
├── train_dense.py     # Trains the dense-crowd point model (GB10)
├── calibrate.py       # Ground calibration helper (click floor points / camera pose)
├── static/
│   ├── index.html     # Live operator dashboard (/ws)
│   ├── twin.html      # 3D digital twin with the 20 s forecast (/ws/twin)
│   └── vendor/three/  # Three.js, served locally
├── weights/           # yolo26strained.pt, yoloheadv26s.pt, p2pnet_crowd_jhu.pth (not in git)
├── videos/            # Demo + Kumbh footage (not in git)
├── calibration/       # auto_<video>.json saved ground estimates
├── dataset/           # GB10: MOT17/, CrowdHuman/, JHU-Crowd/, kumbh_points/ (not in git)
├── scripts/           # gb10_setup.sh, fetch_datasets.py, preflight.py, run_gb10.sh
├── tools/             # annotate_points.py (label heads), iot_gate_sim.py
├── tests/             # runner.py + test_forecast / test_scene / test_dense / test_api
├── experience/        # ExperienceBuffer: per-frame logs + sampled frames
├── reference/         # Training notebooks + capstone report
├── requirements.txt       # Full pipeline (PyTorch + YOLO26)
└── requirements-twin.txt   # Twin-only (no PyTorch)
```

---

## 5. Configuration Parameters

All tuning constants live in the `Config` class at the top of `main.py`; the
forecast physics are in `forecast.ForecastParams`. The main ones:

| Group | Constant | Value | Meaning |
|-------|----------|-------|---------|
| Body model | `YOLO_MODEL` / `_BASE` | `weights/yolo26strained.pt` / `yolo26s.pt` | FBOX detector (+ base fallback) |
| Body model | `YOLO_CONF` / `YOLO_IOU` / `YOLO_IMGSZ` | `0.35` / `0.50` / `960` | Confidence, NMS IoU, input size |
| Body model | `MAX_DET` | `1000` | Per-frame cap (Ultralytics default 300 clips dense crowds) |
| Head model | `HBOX_MODEL` / `HBOX_CONF` / `HBOX_IMGSZ` | `weights/yoloheadv26s.pt` / `0.20` / `1280` | HBOX detector |
| Dense | `HEADS_AS_PEOPLE` | `"dense"` | Head-only people become agents in dense mode only |
| Dense | `DENSE_MIN_HEADS` / `DENSE_HEAD_RATIO` | `40` / `2.0` | Switch on at ≥40 head-only people and ≥2× the bodies |
| Dense | `HBOX_IMGSZ_DENSE` / `HBOX_CONF_DENSE` | `2560` / `0.15` | Head detector in dense mode (3–10 px heads) |
| Dense | `DENSE_MODEL` | `p2pnet_crowd_jhu.pth`, else `_crowd.pth` | Point model, used if present |
| Dense | `DENSE_THRESHOLD` / `DENSE_ENHANCE` | `0.5` / `True` | Head-point score; contrast-equalise first (fog/dusk) |
| Dense | `HEAD_TRACK_SCALE` | `3.0` | Head-only people tracked as a box this many head sizes wide |
| Pipeline | `FRAME_SKIP` | `2` | Process every Nth source frame |
| Pipeline | `FRAME_WIDTH/HEIGHT` | `1280×720` | Max working resolution |
| Pipeline | `STABILIZE` | `True` | Camera-motion compensation (auto-on when the view moves) |
| Calibration | `AUTO_CALIBRATE` / `CAMERA_HFOV_DEG` / `PERSON_HEIGHT_M` | `True` / `65°` / `1.7 m` | Auto ground estimate assumptions |
| Calibration | `SCENE_WIDTH_M × SCENE_HEIGHT_M` | `40 × 22.5 m` | Flat fallback scale |
| Zones | `ZONE_ROWS × ZONE_COLS` | `2×3` | Zone grid |
| Risk | `DENSITY_HIGH` / `SPEED_HIGH` | `0.40 p/m²` / `1.5 m/s` | Risk terms saturate at 2× (0.8 p/m² = Fruin LOS F; 3 m/s) |
| Forecast | `PRED_HORIZON × SIM_STEP_S` | `25 × 0.8 s` = 20 s | Horizon; `SIM_SUBSTEPS` = 4 |
| Forecast | `TRACKER_BUFFER` | `30` | Processed frames a lost track stays in the twin |
| Forecast | `WHATIF_STEPS` | `50` (×0.8 s = 40 s) | Panic what-if horizon |
| Fusion | `W_VIDEO / W_IOT` / `DISCREP_THR` | `0.70 / 0.30` / `5` | Count-fusion weights; discrepancy flag |
| Output | `SEND_FRAME` / `WS_HZ` | `True` / `10` | Video pixels in payload; max broadcasts/s |

Densities and speeds are in **metres** (persons/m², m/s), via the ground
homography — not pixels.

---

## 6. Core Data Structures

**`Agent`** (dataclass) — one tracked person in the twin: `id`, image position
`x, y`, ground position `wx, wy` (metres), ground velocity `vx, vy` (m/s),
`speed`, `vel_std` (velocity uncertainty), `walking` (0 standing … 1 walking),
`zone`, `confidence` (fused belief), `last_seen`, `missed`.

**`GroundPlane`** — the perspective homography from image pixels to ground metres:
`to_world`, `to_image`, `m_per_px`, zone/floor geometry, and `feet_from_heads`
(where a person stands given their head box). Built from a calibration file, an
estimate, or a flat scale (§9).

**`Config`** — every tunable constant (see §5). **`ForecastParams`** (in
`forecast.py`) — the forecast physics (gate, avoidance, Weidmann, arrivals).

**Pipeline payload** (broadcast over `/ws`, cached as `latest_payload`): frame
index, timestamp, `n_agents`, fused count, mean speed/confidence, risk, trend,
`sim_triggered`, `fps`, `latency_ms`, `dense_mode`, `det_counts` (body/head/point),
a `fusion` object, a `zones` list (count/density/risk/flow/predicted), `alerts`,
`bounding_boxes`, the annotated `image_b64`, and an `nfr` pass/fail object.

**Twin snapshot** (read by the 3D twin over `/ws/twin`): crowd state in metres —
ids, positions, velocities, per-agent speed and zone, the learned flow field,
arrivals, floor, obstacles, zone counts/density/risk, trigger and risk.

---

## 7. End-to-End Implementation Walkthrough

Per processed frame (`CDTPipeline.process_frame` → `TwinPipeline.process_tracks`):

1. **Detect** — FBOX (body) + HBOX (head) YOLO26 passes; in dense mode the point
   model too (§8).
2. **Head-only people** — heads with no body box become whole-person detections
   (§8); each body claims at most one head so nobody is counted twice.
3. **Track** — `sv.ByteTrack` gives stable ids. Head-only people are tracked by a
   box sized to their head (`HEAD_TRACK_SCALE`), not a guessed full body.
4. **Camera motion** — `stabilize.CameraMotion` measures people against the ground,
   not the image, and re-anchors after a large pan (`STABILIZE`).
5. **Ground calibration** — on early frames, estimate the ground from box heights
   if no file is given (§9); map each person's feet to metres.
6. **Fusion (A)** — `DataFusionLayer` combines the video count with the IoT gate
   count and rescales detection confidence on discrepancy.
7. **Twin update (G)** — `DigitalTwinEngine.update` folds detections into agents;
   each person's feet go through a **Kalman filter** on the ground
   (`forecast.MotionFilter`) whose measurement noise comes from the perspective,
   so far-away jitter doesn't read as walking. Lost tracks persist for
   `TRACKER_BUFFER` frames.
8. **Scene memory** — `SceneMemory` observes lanes (a flow field), entry points and
   obstacles (floor never walked on).
9. **Density & flow (H)**, **trend (I)**, **risk (L)**, **alerts (M)** — per zone,
   in metres; risk uses each zone's median speed so a few mis-tracks can't max it.
10. **Simulation trigger (J)** — flags the forecast when GROWING, risk MEDIUM/HIGH,
    or fusion confidence κ < 0.70.
11. **Heatmap + payload** — a 1 m Gaussian-per-person density map blended under the
    boxes, JPEG-encoded into the payload, broadcast at up to `WS_HZ`.

The **3D twin** (`TwinService`, §10–11) runs separately: it reads the twin
snapshot and forecasts 20 s ahead once a second while a viewer is open.

### Layer map (report layers A–O)

| Layer | Name | Where |
|-------|------|-------|
| A | Data Fusion | `DataFusionLayer`, `IoTSimulator` |
| B | Edge AI Detection | `CDTPipeline.detect` (FBOX + HBOX), dense mode + `dense.PointCounter` |
| C | ByteTrack MOT | `sv.ByteTrack` |
| D | Confidence Estimator | `ConfidenceEstimator` |
| E | Zone Manager | `ZoneManager` + `GroundPlane` / `PedestrianCalibrator` |
| F | Crowd State | `Agent`, `DigitalTwinEngine.agents` |
| G | Digital Twin Engine | `DigitalTwinEngine` (+ `forecast.MotionFilter`) |
| H | Density & Flow | `DensityFlowEstimator` |
| I | Trend Predictor | `TrendPredictor` |
| J | Simulation Trigger | `SimulationTrigger` |
| K | Short-Horizon Simulation | `TwinService.forecast` + `forecast.py` (`CrowdForecaster`, `SceneMemory`, `ForecastSkill`); `SocialForceModel` is kept as the evaluation baseline |
| L | Risk Estimator | `RiskEstimator` |
| M | Alert Engine | `AlertEngine` |
| N | Quantitative Evaluation | `evaluate.py` |
| O | Dashboard / Operator UI | `static/index.html` (`/ws`) + `static/twin.html` (`/ws/twin`) |

---

## 8. Detection, Dense Mode & the Point Model

The **FBOX** (full-body) detector feeds ByteTrack and the twin. Heads from the
**HBOX** detector with no body box around them are people the body detector
missed: each becomes a whole-person detection — head on top, feet from the ground
geometry (exact with a camera model; `BODY_PER_HEAD` = 7 head-heights otherwise)
— so the twin still places them on the floor. Each body box claims at most one
head (`_heads_of_bodies`), and duplicate points on one head are merged, so nobody
is double-counted.

**Dense mode.** In a dense crowd (a Kumbh ghat) bodies are hidden and heads are
3–10 px, so the box detectors miss most people. When at least `DENSE_MIN_HEADS`
(40) heads have no body **and** they are ≥ `DENSE_HEAD_RATIO` (2×) the bodies, the
pipeline switches to dense mode by itself (and off again below half of each).
Then:

- The head detector runs at `HBOX_IMGSZ_DENSE` (2560 px) for tiny heads.
- If a point model is present, **P2PNet** (`dense.PointCounter`) finds one point
  per head — far more than the box detector in a crowd.
- The frame's contrast is equalised first (`DENSE_ENHANCE`, CLAHE) so heads in fog
  and at dusk stand out.
- Each head point is sized for its image row from the head detector's boxes
  (`dense.HeadScale`, a robust line of head size vs row), not from point spacing —
  which kept the drawn boxes from ballooning where the model missed neighbours.
- Head-only people are tracked by a box `HEAD_TRACK_SCALE` (3×) their head size, so
  the tracker follows the heads, not an overlapping guessed body.

The point model is **P2PNet** (Song et al., ICCV 2021): a VGG16-BN backbone, an
FPN decoder, and regression/classification heads predicting one offset + score per
anchor. `weights/p2pnet_crowd_jhu.pth` (CrowdHuman + JHU-Crowd++) is preferred when
present over `weights/p2pnet_crowd.pth` (CrowdHuman only); on JHU-Crowd++
validation the JHU model counts 82 % of people (MAE 78) vs 43 % (MAE 175) for the
CrowdHuman-only one. Inference is FP16 on CUDA, else CPU.

---

## 9. Perspective Ground Calibration

Each person stands where their feet are (box bottom). `GroundPlane` maps that pixel
to metres with a homography, so zone areas, densities (persons/m²) and speeds
(m/s) account for perspective — in an oblique view a zone near the top of the frame
can cover 10× the floor of one at the bottom. The calibration comes from, in order:

1. **`--calibration file.json`** (`calibrate.py`): ≥4 floor points with known real
   positions, or the camera's height, tilt and field of view.
2. **Automatic estimate** (`PedestrianCalibrator`, single-view metrology): a
   person's pixel height grows linearly with how far below the horizon their feet
   are, giving the camera's height and tilt. It runs on the first ~20–300 frames,
   needs ~300 unobstructed full-body boxes, assumes `CAMERA_HFOV_DEG` (65°) and
   `PERSON_HEIGHT_M` (1.7 m), and is saved to `calibration/auto_<video>.json` for
   reuse. In dense crowds, head boxes do the same job (heads are `HEAD_SIZE_M` =
   0.25 m on a plane 1.45 m up) once they outnumber bodies.
3. **Flat scale** (`SCENE_WIDTH_M × SCENE_HEIGHT_M`, 40 × 22.5 m) when neither is
   available (overhead/drone or very sparse scenes).

The floor's far edge is learned from where feet have been seen (it only moves up);
the image above it (walls, sky) is not counted as floor. A wrong field of view
mostly scales depth (a 50° vs 80° guess changes far-zone areas up to ~2×).

---

## 10. The Forecast (forecast.py)

The 3D twin forecasts `PRED_HORIZON × SIM_STEP_S` = 25 × 0.8 s = 20 s ahead, once
a second, with `CrowdForecaster` — an anticipatory crowd model:

- **Who is walking.** A de-biased speed (noise bias removed) and a significance
  gate decide a 0–1 walking weight (`walking_weight`). The gate requires the speed
  to be clearly above the velocity noise, so a standing crowd's jitter doesn't read
  as walking — tuned (speed ≥ 0.3 m/s, 1.5–2.5 σ) so noisily-measured distant
  walkers aren't frozen either.
- **Where walkers head.** Their own heading, turning towards the lanes the twin has
  learned (`SceneMemory.flow_field`); standing people stay put.
- **Avoidance.** Time-to-collision power law (Karamouzas et al. 2014) with a
  sidestep and keep-right rule — people avoid only those they are about to meet, so
  a dense standing crowd doesn't blow apart. Detours route around learned obstacles.
- **Density.** Walking slows as local density rises (Weidmann's fundamental diagram).
- **Boundaries.** People leave through the edges of the camera's view and slide
  along walls; newcomers arrive where and as often as people have been seen entering
  (`SceneMemory.arrivals`).
- **Self-check.** `ForecastSkill` scores each zone-count forecast against what then
  happened and learns, per horizon, how far to trust the predicted change over "no
  change"; the 3D twin shows that running skill.

On MOT17 held-out ground truth the forecast is competitive with a constant-velocity
baseline (8 s position error ≈ 6.7 m vs 6.4 m) and beats it at mid-horizons; on the
local Kumbh/temple clips it wins clearly. The old Helbing social-force model
(`SocialForceModel`) is kept only as an `evaluate.py` baseline.

**Panic what-if** (`CrowdForecaster.panic`): escape-panic model (Helbing, Farkas &
Vicsek 2000) — everyone flees to the nearest edge of the view, following the crowd;
anticipation fades; bodies compress to contacts; exits pass at most 1.3 people/m/s
(Weidmann/SFPE). It returns every path and each person's local density over time,
flagging where a crush could form. Exposed at `/api/twin/whatif?panic=0.8`.

---

## 11. The Dashboard & 3D Twin

**Operator dashboard** (`static/index.html`, `/ws`): a single-file vanilla HTML/JS
page. Header with a live **source switcher** (the videos under `videos/`, webcam,
RTSP), a metrics row (tracked persons + sparkline, risk score, trend, simulation
status), a left panel (model, tracker, FPS, latency, ground calibration, **dense
mode**, how people were **found** — body/head/point), the live annotated feed with
the 2×3 **zone grid** (count, fill vs capacity, risk, click for details), a right
panel (incident alerts, fusion stream comparison, forecast capacity warnings) and a
ticker. Auto-reconnects.

**3D twin** (`static/twin.html`, `/twin`, `/ws/twin`): Three.js (served locally).
It builds the scene from the calibration — the walkable floor with a 2 m grid, the
zones, **walls** where the floor ends, markers at the **edges of the view**, and
the camera at its estimated height and tilt. People are instanced low-poly proxies
coloured by speed or zone risk. Scrub or play the 0–20 s timeline to see translucent
forecast proxies, their paths, zones coloured by predicted density, and when a zone
reaches capacity. Views: from the real camera (CCTV), 3/4, and top-down. A panic
what-if panel draws the escape paths and crush-risk density.

---

## 12. Web Server & REST/WebSocket API

FastAPI app (`version="2.0"`), open CORS. A module-global `pipeline` holds the
running instance; the asyncio loop is cached at startup for cross-thread broadcast.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ws` | WS | Full dashboard payload at up to 10 Hz |
| `/ws/twin` | WS | 3D twin: scene + crowd state + forecast |
| `/api/snapshot` | GET | Last payload (image stripped) |
| `/api/zones` | GET | Zone geometry + area in m² |
| `/api/fusion_log` | GET | Last 20 fusion results |
| `/api/twin` | GET | 3D twin scene, crowd state, 20 s forecast |
| `/api/twin/whatif?panic=0.8` | GET | Escape-panic what-if |
| `/api/sources` | GET | Videos the dashboard can switch to |
| `/api/source` | POST | Switch source `{"name"}` or `{"url"}` |
| `/api/iot` | POST | Real gate counts `{"entry","exit"}` (optional `X-IoT-Token`) |
| `/api/health` | GET | Video arriving?, uptime, FPS, stream reconnects |
| `/` , `/twin` | GET | The dashboard and the 3D twin pages |

`startup` calls `pipeline.start()` (if one was constructed); `shutdown` stops it.
`ConnectionManager` tracks sockets and prunes dead ones on broadcast.

---

## 13. Datasets, Training & the GB10

The project uses **only MOT17, CrowdHuman and JHU-Crowd++**. All GPU work runs on
the college **GB10** (NVIDIA GB10, ARM64, FP16); the laptop measures on local
videos and replays tracks.

| Use | Train | Test / validation |
|-----|-------|-------------------|
| Body detector (`yolo26strained.pt`) | CrowdHuman + first 85 % of each MOT17 train sequence | CrowdHuman val + last 15 % of each sequence |
| Head detector (`yoloheadv26s.pt`) | CrowdHuman | CrowdHuman val |
| Dense point model (`p2pnet_crowd.pth`) | CrowdHuman train (15,000) | CrowdHuman val |
| Dense point model (`p2pnet_crowd_jhu.pth`) | CrowdHuman + JHU-Crowd++ | CrowdHuman val + JHU-Crowd++ val |
| MOTA + forecast test | — | MOT17 train sequences 02, 04, 09 (static cameras) |

`train_dense.py` reads CrowdHuman `.odgt` head boxes (masking unlabelled crowd
regions so they aren't taught as background), JHU-Crowd++ point files, and your own
`points:` folders (`tools/annotate_points.py`). It trains P2PNet with Hungarian
point matching, early stopping (`--patience`) and resume (`--resume`).

GB10 workflow:

```bash
bash scripts/gb10_setup.sh                 # venv + ARM64 CUDA PyTorch + requirements
python scripts/fetch_datasets.py mot17     # MOT17 02/04/09 via HTTP range requests
python scripts/fetch_datasets.py crowdhuman jhu   # check licence-gated datasets are unpacked
python scripts/preflight.py                # imports, GPU kernels, weights, datasets
bash scripts/run_gb10.sh [measure|mot17|train|jhu|all]
```

`run_gb10.sh`: `measure` runs the speed, MOT17 and crowd-count benchmarks; `mot17`
the held-out tracking; `train` trains the point model on CrowdHuman; `jhu`
fine-tunes it on CrowdHuman + JHU-Crowd++ (`p2pnet_crowd_jhu.pth`) and scores both.
Weights are copied to the laptop's `weights/` by hand (not in git).

---

## 14. Evaluation (Layer N)

```bash
python evaluate.py bench --source videos/demo.mp4 --frames 300          # end-to-end FPS / latency
python evaluate.py tracks --source videos/demo.mp4 --out dataset/tracks/demo   # one pass → MOT tracks
python evaluate.py forecast --tracks <MOT17 seq | tracks folder>        # 20 s forecast vs reality
python evaluate.py mot17 --seq <MOT17>/train/MOT17-09-FRCNN --start-frac 0.85  # count + CLEAR-MOT
python evaluate.py dense --data crowdhuman:dataset/CrowdHuman --max 500 --enhance   # crowd-count error
```

- **`bench`** times the whole pipeline (decode → detect → track → twin → overlay →
  JSON), not just the detector, and reports the NFR pass/fail.
- **`mot17`** processes every frame and scores tracking with CLEAR-MOT (MOTA, MOTP,
  ID switches) plus count error. `--start-frac 0.85` scores only the held-out last
  15 % of a sequence — the frames the body model never trained on — so the numbers
  aren't inflated.
- **`forecast`** replays tracks in time order like the live twin and, every second,
  forecasts with each model — persistence, constant velocity, the old twin, and the
  new twin (with flow/arrivals ablations) — then scores positions (ADE/FDE) and zone
  counts at 2.4–20 s against what actually happened.
- **`dense`** reports count error (MAE/RMSE, share counted) for bodies, bodies +
  heads (normal and dense resolution) and the point model, on CrowdHuman
  (`crowdhuman:`), JHU-Crowd++ (`jhu:`) or your own frames (`points:`); `--enhance`
  applies the pipeline's contrast step; `--weights` scores a specific checkpoint.

Each run prints metrics and saves JSON to `results/`.

---

## 15. Non-Functional Requirements & Performance

Measured on the GB10 (FP16), current code, with the GPU free:

| NFR | Target | Measured |
|-----|--------|----------|
| End-to-end FPS (`demo.mp4`, 1280×596) | ≥ 10 | **10.3 FPS**, p95 latency 107 ms |
| End-to-end latency | < 2000 ms | 107 ms (p95) |
| Dense `kumbhvideo4` (768×432, ~700 points) | — | 4.0 FPS, p95 287 ms |
| MOT17 held-out MOTA (02 / 04 / 09) | — | 0.49 / 0.62 / 0.70 |
| Dense count on CrowdHuman val | — | MAE ≈ 4.0, ~88 % counted (point model) |

The normal pipeline meets the ≥ 10 FPS target; dense mode is heavier (two detectors
at high resolution + the point model) and is the target for TensorRT (`export.py`)
or running the point model every few frames (`DENSE_EVERY`). On a laptop CPU the
full pipeline runs at ~2–3 FPS; twin-only replay handles ~200 people at ~11 ms/frame.

---

## 16. Known Limitations & Caveats

- **Fixed camera assumed** unless motion compensation turns on; it re-anchors after
  large pans.
- **Dense low-res footage undercounts.** On 768×432 Kumbh clips, heads are 2–5 px;
  the point model is well-calibrated in sparse/moderate density but undercounts the
  densest crowds. No ground truth exists for those clips, and reliable per-head
  labelling by eye isn't possible at that resolution — closing the gap needs
  higher-resolution source footage or hand-labelled frames.
- **Feet from heads** use 7 head-heights when there is no camera model.
- **Automatic calibration** assumes a 65° field of view and 1.7 m people, and needs
  unobstructed side-view full-body boxes; it can't estimate overhead/drone views.
- **IoT stream is simulated** unless real gate counts are posted (`/api/iot`) or an
  MQTT broker is attached (`--mqtt`).
- **Panic what-if** bodies are frictionless, so exit capacity is capped by rule
  (1.3 people/m/s).
- **Forecast paths get busy above ~200 people**, and learned arrivals can over-count
  slightly in flickering dense detection.

---

*This document reflects the current code on branch `DTv2`. See `README.md` for the
quick start and the capstone report (`reference/`) for the academic rationale.*
