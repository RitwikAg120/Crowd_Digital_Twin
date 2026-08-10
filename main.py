"""
Crowd Digital Twin — Real-Time Backend
Team 28 | PES University | UE23CS320A Capstone Phase 2
Guide: Dr. Richa Sharma

Members:
  Rishikesh Suraj   (CS481)
  Rithvik Rajesh Matta (CS485)
  Shreyas Sreenivas (CS565)
  Ritwik Agrawal    (CS908)

Architecture:
  CCTV/IoT/Mobility → VideoInputHandler → DataFusionLayer
  → Edge AI (YOLOv8n) → ByteTrack → ConfidenceEstimator
  → CrowdStateRepresentation + Heatmaps → ZoneManager
  → DensityFlowEstimator → DigitalTwinEngine (Agent-Based)
  → TrendPredictor → SimulationTrigger → ShortHorizonSimulation
  → RiskEstimator → AlertEngine → Dashboard → Human Operator

Run:
  pip install -r requirements.txt
  python main.py --source videos/your_video.mp4
  python main.py --source 0                        # webcam
  python main.py --source rtsp://IP:554/stream     # IP camera
"""

import argparse
import asyncio
import base64
import json
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress
from ultralytics import YOLO
import supervision as sv
# Experience buffer for adaptive learning (logs + sampled frames)
from experience.experience_buffer import ExperienceBuffer


# ─── Global Config ────────────────────────────────────────────────────────────

class Config:
    # Model
    YOLO_MODEL        = "weights/yolov8n_cdt.pt"   # fine-tuned; falls back to base
    YOLO_MODEL_BASE   = "yolov8n.pt"
    YOLO_CONF         = 0.35
    YOLO_IOU          = 0.50
    YOLO_IMGSZ        = 640

    # Pipeline
    FRAME_SKIP        = 2          # process every Nth frame
    FRAME_WIDTH       = 1280
    FRAME_HEIGHT      = 720

    # Zones
    ZONE_ROWS         = 2
    ZONE_COLS         = 3

    # Risk thresholds
    DENSITY_LOW       = 0.15       # persons per 10k px²
    DENSITY_HIGH      = 0.40
    VELOCITY_THR      = 25.0       # px/frame — alert threshold
    RISK_WINDOW       = 15         # frames for trend window

    # Simulation
    PRED_HORIZON      = 25         # steps (~20s at 25fps/skip2)
    TRACKER_BUFFER    = 30

    # Fusion weights
    W_VIDEO           = 0.70
    W_IOT             = 0.30
    DISCREP_THR       = 5          # persons before flagging discrepancy

    # IoT simulator
    IOT_NOISE_STD     = 2.0
    IOT_ENTRY_RATE    = 0.10       # fraction of crowd per window
    IOT_EXIT_RATE     = 0.08

    # Density
    HEATMAP_SIGMA     = 30.0

    # WebSocket
    WS_HZ             = 10         # max broadcasts per second

    # ETH/UCY path (optional — for mobility priors)
    ETH_UCY_PATH      = "dataset/eth_ucy"


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class Agent:
    id:         int
    x:          float
    y:          float
    vx:         float = 0.0
    vy:         float = 0.0
    speed:      float = 0.0
    confidence: float = 1.0
    zone:       str   = ""
    history:    list  = field(default_factory=list)


@dataclass
class ZoneState:
    name:     str
    x1:       int
    y1:       int
    x2:       int
    y2:       int
    capacity: int = 50

    @property
    def area(self) -> float:
        return float((self.x2 - self.x1) * (self.y2 - self.y1))

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2


# ─── Zone Manager ─────────────────────────────────────────────────────────────

class ZoneManager:
    def __init__(self, h: int, w: int, rows: int = Config.ZONE_ROWS,
                 cols: int = Config.ZONE_COLS):
        self.zones: Dict[str, ZoneState] = {}
        rh, cw = h // rows, w // cols
        labels = "ABCDEFGHIJKLMNOP"
        idx = 0
        for r in range(rows):
            for c in range(cols):
                name = f"Zone_{labels[idx]}"
                self.zones[name] = ZoneState(
                    name=name,
                    x1=c * cw,
                    y1=r * rh,
                    x2=(c + 1) * cw if c < cols - 1 else w,
                    y2=(r + 1) * rh if r < rows - 1 else h,
                    capacity=max(10, int((cw * rh) / (w * h) * 150))
                )
                idx += 1

    def assign(self, cx: float, cy: float) -> str:
        for name, z in self.zones.items():
            if z.x1 <= cx < z.x2 and z.y1 <= cy < z.y2:
                return name
        return "Zone_X"

    def count(self, agents: List[Agent]) -> Dict[str, int]:
        c = {n: 0 for n in self.zones}
        for a in agents:
            if a.zone in c:
                c[a.zone] += 1
        return c


# ─── Confidence Estimator ─────────────────────────────────────────────────────

class ConfidenceEstimator:
    """
    Per-agent confidence = detection_conf × age_weight × zone_calibration_weight
    Addresses the gap noted in Phase 2 review.
    """
    def __init__(self):
        self._age: Dict[int, int] = defaultdict(int)
        self._zone_cal: Dict[str, float] = defaultdict(lambda: 1.0)

    def update(self, agent_id: int, det_conf: float,
               zone: str, zone_density: float) -> float:
        self._age[agent_id] += 1
        age_w  = min(1.0, self._age[agent_id] / 5.0)   # ramp up over 5 frames
        zone_w = self._zone_cal.get(zone, 1.0)
        dens_w = max(0.6, 1.0 - zone_density * 0.4)    # dense zones → lower conf
        return round(float(det_conf) * age_w * zone_w * dens_w, 3)

    def set_zone_calibration(self, zone: str, weight: float):
        """Set from ETH/UCY mobility prior or manual calibration."""
        self._zone_cal[zone] = max(0.5, min(1.0, weight))

    def drop(self, agent_id: int):
        self._age.pop(agent_id, None)


# ─── IoT Simulator ────────────────────────────────────────────────────────────

class IoTSimulator:
    """
    Simulates physical entry/exit gate sensors.
    In real deployment: replace simulate_tick() with MQTT subscriber.
      e.g.  import paho.mqtt.client as mqtt
            client.subscribe("venue/gate/entry")
            def on_message(msg): self.push_real(int(msg.payload), 0)
    """
    def __init__(self, noise_std: float = Config.IOT_NOISE_STD):
        self.noise_std          = noise_std
        self._cumulative_entry  = 0
        self._cumulative_exit   = 0
        self._history           = deque(maxlen=200)

    def simulate_tick(self, video_count: int, frame_idx: int) -> Tuple[int, int]:
        noise  = np.random.normal(0, self.noise_std)
        entry  = max(0, int(video_count * Config.IOT_ENTRY_RATE + noise))
        exit_  = max(0, int(video_count * Config.IOT_EXIT_RATE  + abs(noise * 0.5)))
        self._cumulative_entry += entry
        self._cumulative_exit  += exit_
        self._history.append((frame_idx, entry, exit_))
        return entry, exit_

    def push_real(self, entry: int, exit_: int):
        """Call this from your real sensor callback instead of simulate_tick."""
        self._cumulative_entry += entry
        self._cumulative_exit  += exit_

    @property
    def net_count(self) -> int:
        return max(0, self._cumulative_entry - self._cumulative_exit)


# ─── Data Fusion Layer ────────────────────────────────────────────────────────

class DataFusionLayer:
    """
    Multi-modal sensor fusion:
      Stream 1: Video detections (YOLOv8n + ByteTrack)   weight=0.70
      Stream 2: IoT gate counters (entry/exit cumulative) weight=0.30
      Stream 3: ETH/UCY mobility priors (zone confidence modifier)

    Key insight: video undercounts in crowds (occlusion).
    IoT overcounts slightly (re-entry, sensor noise).
    Weighted fusion + discrepancy detection corrects both.
    """

    def __init__(self):
        self.iot      = IoTSimulator()
        self.mobility: Dict[str, float] = {}   # zone_name -> confidence_weight
        self._log     = deque(maxlen=300)

    def load_mobility_prior(self, eth_ucy_root: str, zone_mgr: ZoneManager):
        """Load ETH/UCY trajectory data to compute per-zone flow priors."""
        eth_path = Path(eth_ucy_root)
        if not eth_path.exists():
            print(f"[Fusion] ETH/UCY path not found: {eth_ucy_root}. Using uniform priors.")
            for name in zone_mgr.zones:
                self.mobility[name] = 1.0
            return

        try:
            import pandas as pd
            all_speeds = []
            for f in eth_path.glob("**/*.txt"):
                try:
                    df = pd.read_csv(str(f), sep="\t", header=None,
                                     names=["frame", "pid", "x", "y"])
                    df = df.sort_values(["pid", "frame"])
                    for _, g in df.groupby("pid"):
                        coords = g[["x", "y"]].values
                        if len(coords) > 1:
                            spd = np.linalg.norm(np.diff(coords, axis=0), axis=1)
                            all_speeds.extend(spd.tolist())
                except Exception:
                    continue

            global_mean = float(np.mean(all_speeds)) if all_speeds else 1.0
            for name in zone_mgr.zones:
                # High global flow → slightly lower zone confidence
                # (extend with per-zone matching if you have zone-labelled data)
                self.mobility[name] = max(0.75, 1.0 - (global_mean / 100.0))

            print(f"[Fusion] ETH/UCY priors loaded. Global mean speed: {global_mean:.3f}")
        except Exception as e:
            print(f"[Fusion] ETH/UCY load failed: {e}. Using uniform priors.")
            for name in zone_mgr.zones:
                self.mobility[name] = 1.0

    def fuse(self, video_detections: list, frame_idx: int) -> dict:
        """
        Call every frame. Returns fusion result dict consumed by DT engine.

        Paper equation:
          C_f = W_VIDEO × C_video + W_IOT × C_iot
          κ   = 1 − (|C_video − C_iot| / max(C_video, C_iot, 1))
        """
        video_count = len(video_detections)

        # Stream 2: IoT tick
        entry, exit_ = self.iot.simulate_tick(video_count, frame_idx)
        iot_count    = self.iot.net_count

        # Discrepancy detection
        discrepancy      = abs(video_count - iot_count)
        discrepancy_flag = discrepancy > Config.DISCREP_THR

        # Confidence rescaling direction
        if discrepancy_flag:
            if iot_count > video_count:
                # IoT sees more → occlusion → boost conf
                conf_scale = min(1.0, 1.0 + discrepancy * 0.02)
            else:
                # Video sees more → false dets → lower conf
                conf_scale = max(0.6, 1.0 - discrepancy * 0.02)
        else:
            conf_scale = 1.0

        # Rescale detection confidences (Stream 1 + Stream 3)
        fused_dets = []
        for det in video_detections:
            zone_w  = self.mobility.get(det.get("zone", ""), 1.0)
            new_det = dict(det)
            new_det["confidence"] = round(
                min(1.0, det["confidence"] * conf_scale * zone_w), 3)
            fused_dets.append(new_det)

        # Weighted count fusion
        fused_count = Config.W_VIDEO * video_count + Config.W_IOT * iot_count

        # Fusion confidence: how much do streams agree?
        max_c        = max(video_count, iot_count, 1)
        fusion_conf  = round(max(0.0, 1.0 - discrepancy / max_c), 3)

        result = {
            "fused_detections":  fused_dets,
            "fused_count":       round(fused_count, 1),
            "video_count":       video_count,
            "iot_count":         iot_count,
            "iot_entry_delta":   entry,
            "iot_exit_delta":    exit_,
            "discrepancy":       discrepancy,
            "discrepancy_flag":  discrepancy_flag,
            "conf_scale":        round(conf_scale, 3),
            "fusion_confidence": fusion_conf,
        }
        self._log.append(result)
        return result


# ─── Digital Twin Engine ──────────────────────────────────────────────────────

class DigitalTwinEngine:
    ALPHA = 0.4   # EMA velocity smoothing

    def __init__(self, zone_mgr: ZoneManager, conf_est: ConfidenceEstimator):
        self.agents:    Dict[int, Agent] = {}
        self.zone_mgr   = zone_mgr
        self.conf_est   = conf_est
        self.frame_idx  = 0
        self.state_log  = deque(maxlen=Config.RISK_WINDOW * 4)

    def update(self, detections: list, density_per_zone: Dict[str, float],
               fused_count: float = 0.0):
        self.frame_idx += 1
        seen = set()

        for det in detections:
            aid   = det["id"]
            xn, yn = det["cx"], det["cy"]
            seen.add(aid)
            zone  = self.zone_mgr.assign(xn, yn)
            zd    = density_per_zone.get(zone, 0.0)
            conf  = self.conf_est.update(aid, det["confidence"], zone, zd)

            if aid in self.agents:
                a  = self.agents[aid]
                vx = self.ALPHA * (xn - a.x) + (1 - self.ALPHA) * a.vx
                vy = self.ALPHA * (yn - a.y) + (1 - self.ALPHA) * a.vy
                a.x, a.y   = xn, yn
                a.vx, a.vy = vx, vy
                a.speed    = math.hypot(vx, vy)
                a.zone     = zone
                a.confidence = conf
                a.history.append((xn, yn))
                if len(a.history) > 12:
                    a.history.pop(0)
            else:
                self.agents[aid] = Agent(
                    id=aid, x=xn, y=yn,
                    zone=zone, confidence=conf,
                    history=[(xn, yn)]
                )

        # Remove stale tracks
        for stale in set(self.agents) - seen:
            self.conf_est.drop(stale)
            del self.agents[stale]

        self.state_log.append({
            "frame":      self.frame_idx,
            "n_agents":   len(self.agents),
            "fused_count": fused_count,
            "mean_speed": float(np.mean([a.speed for a in self.agents.values()])
                                if self.agents else 0.0),
        })

    def simulate_future(self, steps: int, h: int, w: int) -> List[Dict]:
        """Short-horizon what-if simulation with boundary reflection."""
        state = {aid: (a.x, a.y, a.vx, a.vy) for aid, a in self.agents.items()}
        preds = []
        for _ in range(steps):
            step = {}
            nxt  = {}
            for aid, (x, y, vx, vy) in state.items():
                nx  = float(np.clip(x + vx, 0, w))
                ny  = float(np.clip(y + vy, 0, h))
                nvx = -vx if nx <= 0 or nx >= w else vx
                nvy = -vy if ny <= 0 or ny >= h else vy
                step[aid] = (nx, ny)
                nxt[aid]  = (nx, ny, nvx, nvy)
            preds.append(step)
            state = nxt
        return preds


# ─── Density & Flow Estimator ─────────────────────────────────────────────────

class DensityFlowEstimator:
    def __init__(self, zone_mgr: ZoneManager):
        self.zone_mgr = zone_mgr

    def density_map(self, agents: List[Agent], h: int, w: int) -> np.ndarray:
        d = np.zeros((h, w), dtype=np.float32)
        for a in agents:
            cx = int(np.clip(a.x, 0, w - 1))
            cy = int(np.clip(a.y, 0, h - 1))
            d[cy, cx] += 1.0
        d = gaussian_filter(d, sigma=Config.HEATMAP_SIGMA)
        if d.max() > 0:
            d /= d.max()
        return d

    def zone_density(self, agents: List[Agent]) -> Dict[str, float]:
        counts = self.zone_mgr.count(agents)
        return {
            name: counts[name] / max(z.area / 1e4, 1.0)
            for name, z in self.zone_mgr.zones.items()
        }

    def zone_flow(self, agents: List[Agent]) -> Dict[str, Tuple[float, float]]:
        vx = defaultdict(list)
        vy = defaultdict(list)
        for a in agents:
            vx[a.zone].append(a.vx)
            vy[a.zone].append(a.vy)
        return {
            n: (float(np.mean(vx.get(n, [0]))),
                float(np.mean(vy.get(n, [0]))))
            for n in self.zone_mgr.zones
        }


# ─── Trend Predictor ─────────────────────────────────────────────────────────

class TrendPredictor:
    SLOPE_THR = 0.15

    def __init__(self, window: int = Config.RISK_WINDOW):
        self.window  = window
        self._counts = deque(maxlen=window)
        self._speeds = deque(maxlen=window)

    def push(self, n: int, spd: float):
        self._counts.append(float(n))
        self._speeds.append(float(spd))

    def analyze(self) -> dict:
        c = list(self._counts)
        if len(c) < 3:
            return {"crowd_trend": "STABLE", "slope": 0.0, "r2": 0.0}
        x = np.arange(len(c), dtype=float)
        sl, _, r, _, _ = linregress(x, c)
        if sl > self.SLOPE_THR:
            trend = "GROWING"
        elif sl < -self.SLOPE_THR:
            trend = "DISPERSING"
        else:
            trend = "STABLE"
        return {
            "crowd_trend": trend,
            "slope":       round(float(sl), 4),
            "r2":          round(float(r ** 2), 3),
        }


# ─── Risk Estimator ───────────────────────────────────────────────────────────

class RiskEstimator:
    """
    Hybrid confidence-weighted risk score.
    Paper formula:
      R = 0.40×density_norm + 0.30×speed_norm + 0.20×trend_score + 0.10×(1−confidence)
    """
    W = {"density": 0.40, "speed": 0.30, "trend": 0.20, "confidence": 0.10}
    MEDIUM_THR = 0.33
    HIGH_THR   = 0.66

    def classify(self, density: float, speed: float,
                 trend: str, confidence: float = 1.0) -> dict:
        d  = float(np.clip(density / (Config.DENSITY_HIGH * 2), 0, 1))
        s  = float(np.clip(speed   / (Config.VELOCITY_THR  * 2), 0, 1))
        t  = {"GROWING": 1.0, "STABLE": 0.3, "DISPERSING": 0.0}.get(trend, 0.3)
        ci = 1.0 - float(np.clip(confidence, 0, 1))
        score = (self.W["density"]    * d  +
                 self.W["speed"]      * s  +
                 self.W["trend"]      * t  +
                 self.W["confidence"] * ci)
        label = ("HIGH"   if score > self.HIGH_THR  else
                 "MEDIUM" if score > self.MEDIUM_THR else "LOW")
        return {"risk_score": round(float(score), 4), "risk_label": label}


# ─── Alert Engine ─────────────────────────────────────────────────────────────

class AlertEngine:
    MSGS = {
        "HIGH":   "CRITICAL: {zone} — {count} agents, HIGH density. Immediate intervention required.",
        "MEDIUM": "WARNING:  {zone} — {count} agents approaching threshold.",
    }

    def generate(self, zone_risks: dict, zone_counts: dict) -> List[str]:
        alerts = []
        for zone, r in zone_risks.items():
            if r["risk_label"] in ("HIGH", "MEDIUM"):
                alerts.append(self.MSGS[r["risk_label"]].format(
                    zone=zone, count=zone_counts.get(zone, 0)
                ))
        return alerts


# ─── Video Input Handler ──────────────────────────────────────────────────────

class VideoInputHandler:
    """
    Thread-safe video capture. Reads from webcam, RTSP, or mp4.
    Auto-loops recorded files for pseudo-live demo mode.
    """
    def __init__(self, source):
        self.source   = source
        self.cap:     Optional[cv2.VideoCapture] = None
        self._frame:  Optional[np.ndarray] = None
        self._lock    = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[Video] Source opened: {self.source}")

    def _capture_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                # Loop recorded video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            with self._lock:
                self._frame = frame

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def resolution(self) -> Tuple[int, int]:
        if self.cap is None:
            return Config.FRAME_HEIGHT, Config.FRAME_WIDTH
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return h, w

    def stop(self):
        self._running = False
        if self.cap:
            self.cap.release()


# ─── Main CDT Pipeline ────────────────────────────────────────────────────────

class CDTPipeline:
    def __init__(self, source):
        self.video   = VideoInputHandler(source)

        # Load fine-tuned weights if available, else fall back to base
        model_path = Config.YOLO_MODEL
        if not Path(model_path).exists():
            print(f"[CDT] Fine-tuned weights not found at {model_path}. Using base YOLOv8n.")
            model_path = Config.YOLO_MODEL_BASE
        self.model   = YOLO(model_path)

        self.tracker = sv.ByteTrack(
            track_activation_threshold=Config.YOLO_CONF,
            lost_track_buffer=Config.TRACKER_BUFFER,
            minimum_matching_threshold=0.8,
            frame_rate=25,
        )

        self._h = Config.FRAME_HEIGHT
        self._w = Config.FRAME_WIDTH

        self.zone_mgr  = ZoneManager(self._h, self._w)
        self.conf_est  = ConfidenceEstimator()
        self.fusion    = DataFusionLayer()
        self.dt        = DigitalTwinEngine(self.zone_mgr, self.conf_est)
        self.density   = DensityFlowEstimator(self.zone_mgr)
        self.trend     = TrendPredictor()
        self.risk_est  = RiskEstimator()
        self.alerts    = AlertEngine()

        # Experience buffer: collects lightweight examples and sampled frames
        # for offline retraining / curation.
        try:
            self.experience = ExperienceBuffer(
                base_path="experience",
                max_buffer=800,
                save_frames=True,
                frame_save_interval=30,
            )
        except Exception:
            self.experience = None

        self._frame_n  = 0
        self._fps_buf  = deque(maxlen=30)
        self._running  = False
        self.latest_payload: Optional[dict] = None

    def start(self):
        self.video.start()
        self._h, self._w = self.video.resolution
        # Rebuild with actual resolution
        self.zone_mgr        = ZoneManager(self._h, self._w)
        self.dt.zone_mgr     = self.zone_mgr
        self.density.zone_mgr = self.zone_mgr
        # Load ETH/UCY mobility priors
        self.fusion.load_mobility_prior(Config.ETH_UCY_PATH, self.zone_mgr)
        # Set zone calibration from mobility priors
        for zone, w in self.fusion.mobility.items():
            self.conf_est.set_zone_calibration(zone, w)
        self._running = True
        threading.Thread(target=self._pipeline_loop, daemon=True).start()
        print("[CDT] Pipeline running.")

    def _pipeline_loop(self):
        last_broadcast    = time.time()
        broadcast_interval = 1.0 / Config.WS_HZ

        while self._running:
            t0    = time.perf_counter()
            frame = self.video.read()
            if frame is None:
                time.sleep(0.01)
                continue

            self._frame_n += 1
            if self._frame_n % Config.FRAME_SKIP != 0:
                continue

            # ── Resize ────────────────────────────────────────────────────────
            frame = cv2.resize(frame, (self._w, self._h))

            # ── Edge AI: YOLOv8n ──────────────────────────────────────────────
            results = self.model(
                frame,
                classes=[0],
                conf=Config.YOLO_CONF,
                iou=Config.YOLO_IOU,
                imgsz=Config.YOLO_IMGSZ,
                verbose=False,
            )
            dets = sv.Detections.from_ultralytics(results[0])

            # ── ByteTrack ─────────────────────────────────────────────────────
            tracked = self.tracker.update_with_detections(dets)

            det_list = []
            if tracked.tracker_id is not None:
                for xyxy, tid, conf in zip(
                    tracked.xyxy, tracked.tracker_id, tracked.confidence
                ):
                    cx = float((xyxy[0] + xyxy[2]) / 2)
                    cy = float((xyxy[1] + xyxy[3]) / 2)
                    det_list.append({
                        "id":         int(tid),
                        "cx":         cx,
                        "cy":         cy,
                        "confidence": float(conf),
                        "zone":       self.zone_mgr.assign(cx, cy),
                    })

            # ── Data Fusion ───────────────────────────────────────────────────
            fusion_result = self.fusion.fuse(det_list, self._frame_n)
            det_list      = fusion_result["fused_detections"]
            fused_count   = fusion_result["fused_count"]

            # ── DT Engine update ──────────────────────────────────────────────
            z_dens_pre = self.density.zone_density(list(self.dt.agents.values()))
            self.dt.update(det_list, z_dens_pre, fused_count=fused_count)

            # ── Post-update state ─────────────────────────────────────────────
            agents_now = list(self.dt.agents.values())
            z_dens     = self.density.zone_density(agents_now)
            z_counts   = self.zone_mgr.count(agents_now)
            z_flow     = self.density.zone_flow(agents_now)

            mean_spd  = float(np.mean([a.speed      for a in agents_now])) if agents_now else 0.0
            mean_conf = float(np.mean([a.confidence for a in agents_now])) if agents_now else 1.0

            # ── Trend predictor ───────────────────────────────────────────────
            self.trend.push(len(agents_now), mean_spd)
            trend_res = self.trend.analyze()

            # ── Risk estimator ────────────────────────────────────────────────
            global_dens = float(np.mean(list(z_dens.values()))) if z_dens else 0.0
            risk_res = self.risk_est.classify(
                global_dens, mean_spd, trend_res["crowd_trend"], mean_conf
            )

            zone_risks = {}
            for zn in self.zone_mgr.zones:
                zs = [a.speed      for a in agents_now if a.zone == zn]
                zc = [a.confidence for a in agents_now if a.zone == zn]
                zone_risks[zn] = self.risk_est.classify(
                    z_dens.get(zn, 0.0),
                    float(np.mean(zs)) if zs else 0.0,
                    trend_res["crowd_trend"],
                    float(np.mean(zc)) if zc else 1.0,
                )

            # ── Alert engine ──────────────────────────────────────────────────
            alert_msgs = self.alerts.generate(zone_risks, z_counts)

            # ── Short-horizon simulation ──────────────────────────────────────
            sim_triggered = (
                trend_res["crowd_trend"] in ("GROWING",) or
                risk_res["risk_label"]   in ("HIGH", "MEDIUM")
            )
            pred_zones: Dict[str, int] = {n: 0 for n in self.zone_mgr.zones}
            if sim_triggered:
                future = self.dt.simulate_future(Config.PRED_HORIZON, self._h, self._w)
                if future:
                    for x, y in future[-1].values():
                        z = self.zone_mgr.assign(x, y)
                        if z in pred_zones:
                            pred_zones[z] += 1

            # ── Heatmap JPEG ──────────────────────────────────────────────────
            dmap       = self.density.density_map(agents_now, self._h, self._w)
            heat       = (dmap * 255).astype(np.uint8)
            heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            overlay    = cv2.addWeighted(frame, 0.55, heat_color, 0.45, 0)

            for z in self.zone_mgr.zones.values():
                rl  = zone_risks[z.name]["risk_label"]
                col = {"HIGH": (50, 50, 220),
                        "MEDIUM": (0, 165, 255),
                        "LOW":    (80, 200, 80)}.get(rl, (128, 128, 128))
                cv2.rectangle(overlay, (z.x1, z.y1), (z.x2, z.y2), col, 2)
                cv2.putText(overlay, f"{z.name}: {rl}",
                            (z.x1 + 6, z.y1 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            for a in agents_now:
                cv2.circle(overlay, (int(a.x), int(a.y)), 4, (255, 255, 100), -1)

            _, buf       = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 72])
            heatmap_b64  = base64.b64encode(buf).decode()

            # ── FPS ───────────────────────────────────────────────────────────
            t1 = time.perf_counter()
            self._fps_buf.append(1.0 / max(t1 - t0, 1e-9))
            fps = round(float(np.mean(self._fps_buf)), 1)
            lat = round((t1 - t0) * 1000, 1)

            # ── WebSocket payload ─────────────────────────────────────────────
            payload = {
                "frame_idx":        self._frame_n,
                "timestamp":        round(time.time(), 3),
                "n_agents":         len(agents_now),
                "fused_count":      round(fused_count, 1),
                "mean_speed":       round(mean_spd, 2),
                "mean_confidence":  round(mean_conf, 3),
                "risk_score":       risk_res["risk_score"],
                "risk_label":       risk_res["risk_label"],
                "crowd_trend":      trend_res["crowd_trend"],
                "trend_slope":      trend_res["slope"],
                "trend_r2":         trend_res["r2"],
                "sim_triggered":    sim_triggered,
                "fps":              fps,
                "latency_ms":       lat,
                "fusion": {
                    "video_count":    fusion_result["video_count"],
                    "iot_count":      fusion_result["iot_count"],
                    "fused_count":    round(fused_count, 1),
                    "confidence":     fusion_result["fusion_confidence"],
                    "discrepancy":    fusion_result["discrepancy"],
                    "occlusion_flag": fusion_result["discrepancy_flag"],
                    "iot_entry":      fusion_result["iot_entry_delta"],
                    "iot_exit":       fusion_result["iot_exit_delta"],
                    "conf_scale":     fusion_result["conf_scale"],
                },
                "zones": {
                    n: {
                        "count":      z_counts.get(n, 0),
                        "density":    round(z_dens.get(n, 0.0), 4),
                        "risk":       zone_risks[n]["risk_label"],
                        "risk_score": zone_risks[n]["risk_score"],
                        "flow_x":     round(z_flow.get(n, (0, 0))[0], 2),
                        "flow_y":     round(z_flow.get(n, (0, 0))[1], 2),
                        "predicted":  pred_zones.get(n, 0),
                        "capacity":   self.zone_mgr.zones[n].capacity,
                    }
                    for n in self.zone_mgr.zones
                },
                "alerts": alert_msgs[:5],
                "agents": [
                    {
                        "id":    a.id,
                        "x":     round(a.x, 1),
                        "y":     round(a.y, 1),
                        "speed": round(a.speed, 2),
                        "zone":  a.zone,
                        "conf":  round(a.confidence, 3),
                    }
                    for a in agents_now
                ],
                "heatmap_b64": heatmap_b64,
                "nfr": {
                    "fps_pass":     fps >= 10,
                    "latency_pass": lat < 2000,
                },
            }
            self.latest_payload = payload

            # Record experience for offline curation / retraining (non-blocking)
            try:
                if getattr(self, "experience", None) is not None:
                    # Pass the raw frame (buffer handles resizing/saving)
                    self.experience.record(self._frame_n, frame, det_list, agents_now, payload)
                    # Flush in background when buffer grows
                    if self.experience.buffer_len() >= 100:
                        threading.Thread(target=self.experience.flush, daemon=True).start()
            except Exception as e:
                print(f"[Experience] record failed: {e}")

            # Rate-limited broadcast
            now = time.time()
            if now - last_broadcast >= broadcast_interval:
                asyncio.run_coroutine_threadsafe(
                    manager.broadcast(json.dumps(payload)),
                    app_loop,
                )
                last_broadcast = now

    def stop(self):
        self._running = False
        self.video.stop()


# ─── WebSocket Manager ────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        print(f"[WS] Client connected. Total: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        print(f"[WS] Client disconnected. Total: {len(self.active)}")

    async def broadcast(self, message: str):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self.active:
                self.active.remove(ws)


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app      = FastAPI(title="Crowd Digital Twin", version="2.0")
manager  = ConnectionManager()
pipeline: Optional[CDTPipeline] = None
app_loop: Optional[asyncio.AbstractEventLoop] = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global app_loop
    app_loop = asyncio.get_event_loop()
    if pipeline:
        pipeline.start()


@app.on_event("shutdown")
async def shutdown():
    if pipeline:
        pipeline.stop()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


@app.get("/api/snapshot")
async def snapshot():
    """REST fallback — latest pipeline state (no heatmap)."""
    if pipeline and pipeline.latest_payload:
        p = dict(pipeline.latest_payload)
        p.pop("heatmap_b64", None)
        return p
    return {"status": "pipeline not started"}


@app.get("/api/zones")
async def zones_info():
    if pipeline:
        return {
            n: {
                "x1": z.x1, "y1": z.y1,
                "x2": z.x2, "y2": z.y2,
                "capacity": z.capacity,
            }
            for n, z in pipeline.zone_mgr.zones.items()
        }
    return {}


@app.get("/api/fusion_log")
async def fusion_log():
    """Return last 20 fusion results for debugging."""
    if pipeline:
        return list(pipeline.fusion._log)[-20:]
    return []


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    try:
        return open("static/index.html").read()
    except FileNotFoundError:
        return "<h1>Dashboard not found. Put static/index.html in place.</h1>"


try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    pass


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crowd Digital Twin — Real-Time Server")
    parser.add_argument(
        "--source", default="videos/mot17_demo.mp4",
        help="Video source: 0=webcam | rtsp://... | path/to/video.mp4"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    pipeline = CDTPipeline(src)

    print("=" * 60)
    print("  Crowd Digital Twin — Team 28 | PES University")
    print(f"  Dashboard  → http://localhost:{args.port}/")
    print(f"  WebSocket  → ws://localhost:{args.port}/ws")
    print(f"  REST API   → http://localhost:{args.port}/api/snapshot")
    print(f"  Source     → {src}")
    print("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
