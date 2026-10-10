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
  CCTV/IoT → VideoInputHandler → DataFusionLayer
  → Edge AI (YOLO26s body + head detectors) → ByteTrack → ConfidenceEstimator
  → CrowdStateRepresentation + Heatmaps → ZoneManager
  → DensityFlowEstimator → DigitalTwinEngine (Agent-Based)
  → TrendPredictor → SimulationTrigger → ShortHorizonSimulation (social force)
  → RiskEstimator → AlertEngine → Dashboard → Human Operator

Run:
  pip install -r requirements.txt
  python main.py --source videos/your_video.mp4
  python main.py --source 0                        # webcam
  python main.py --source rtsp://IP:554/stream     # IP camera
  python evaluate.py --help                        # Layer N: quantitative evaluation
"""

import argparse
import asyncio
import base64
import json
import math
import os
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import uvicorn
from fastapi import Body, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.stats import linregress
# Experience buffer for adaptive learning (logs + sampled frames)
from experience.experience_buffer import ExperienceBuffer
# Gate counters (Stream 2) over HTTP / MQTT
from iot import MQTTGates, parse_counts
from multicam import SiteFusion
# Camera-motion compensation (handheld / panning footage)
from stabilize import CameraMotion, apply as apply_transform
# Motion state and the 20 s forecast (Layer K)
from forecast import (CrowdForecaster, Floor, ForecastParams, ForecastSkill, MotionFilter,
                      SceneMemory, walking_weight)


# ─── Global Config ────────────────────────────────────────────────────────────

class Config:
    # Model
    YOLO_MODEL        = "weights/yolo26strained.pt"   # fine-tuned; falls back to base
    YOLO_MODEL_BASE   = "yolo26s.pt"
    YOLO_CONF         = 0.35
    YOLO_IOU          = 0.50
    YOLO_IMGSZ        = 960
    MAX_DET           = 1000       # detections per frame (Ultralytics default of 300 caps dense crowds)
    DEVICE            = "auto"     # "auto" → CUDA when available, else CPU
    HALF              = True       # FP16 inference on CUDA

    HBOX_MODEL        = "weights/yoloheadv26s.pt"
    HBOX_CONF         = 0.20
    HBOX_IMGSZ        = 1280
    HBOX_EVERY        = 1          # run the head detector every Nth processed frame

    # Dense crowds: people seen only as heads. Heads (or head points) with no
    # body box become people; when many are, dense mode looks harder.
    # People seen only as a head: "dense" = in dense mode only. In normal crowds
    # their guessed whole-body boxes add no one the body detector misses on
    # MOT17 and cost MOTA (held-out 02 / 04 / 09: 0.48 / 0.54 / 0.47 with them,
    # 0.49 / 0.62 / 0.70 without). True = always, False = never.
    HEADS_AS_PEOPLE   = "dense"
    HEAD_SIZE_M       = 0.25       # head box height: the ruler when calibrating from heads
    BODY_PER_HEAD     = 7.0        # body height in head heights, without a camera model
    DENSE_AUTO        = True       # switch dense mode on/off by itself
    # A dense crowd is mostly heads: MOT17-02 (a busy street, ~30 people) shows
    # 15–35 heads without a body against ~20 bodies; the Kumbh clips 120–450
    # against 6–22. Off again below half of both.
    DENSE_MIN_HEADS   = 40         # on when ≥ this many heads have no body …
    DENSE_HEAD_RATIO  = 2.0        # … and they are ≥ this many times the bodies
    HBOX_IMGSZ_DENSE  = 2560       # head detector input in dense mode (3–10 px heads)
    HBOX_CONF_DENSE   = 0.15
    # Head-point model (dense.py), used if present, best first: the Kumbh-adapted
    # model (CrowdHuman + JHU-Crowd++ + labelled Kumbh frames), then the JHU one,
    # then CrowdHuman-only. On the same gold JHU+CrowdHuman val, the Kumbh model
    # beats the JHU one (overall MAE 8.7 vs 9.5, JHU counted 94% vs 89%).
    DENSE_MODEL       = next((w for w in ("weights/p2pnet_crowd_kumbh.pth",
                                          "weights/p2pnet_crowd_jhu.pth",
                                          "weights/p2pnet_crowd.pth") if Path(w).exists()),
                             "weights/p2pnet_crowd.pth")
    DENSE_THRESHOLD   = 0.5        # head score for a point
    DENSE_ENHANCE     = True       # equalise contrast before the point model (fog, dusk)
    DENSE_EVERY       = 1          # run the point model every Nth processed frame in dense mode
    DENSE_ALTERNATE   = True       # dense mode: 2560 px heads on even frames, head points on odd
                                   # ones (each reuses the other's last result) — ~1 heavy model less
                                   # per frame; packed crowds move only a few px between frames
    DENSE_PROBE_EVERY = 50         # with the point model: look for a dense crowd every Nth frame
    HEAD_TRACK_SCALE  = 3.0        # heads are tracked as boxes this many head sizes wide
                                   # even when the head detector sees none (drone / overhead views)
    CALIB_MIN_HEADS   = 600        # head boxes to calibrate from, when bodies are too few

    # Pipeline
    FRAME_SKIP        = 2          # process every Nth source frame; 1 = every frame
    FRAME_WIDTH       = 1280       # max working resolution; larger sources are downscaled
    FRAME_HEIGHT      = 720
    SOURCE_FPS        = 25.0       # used when the source does not report its frame rate
    STREAM_RETRY_READS = 200       # failed reads (~2 s) before a live stream is reopened
    STABILIZE         = True       # compensate camera motion (turns itself on only when the view moves)

    # Ground-plane calibration (perspective). Pixels map to metres on the ground
    # through a homography taken from, in order: CALIBRATION_FILE, an estimate
    # from people's box heights (AUTO_CALIBRATE), or the flat SCENE_* scale.
    CALIBRATION_FILE  = None       # JSON: ≥4 image↔ground points, or camera height/pitch/FOV
    AUTO_CALIBRATE    = True
    REUSE_AUTO_CALIBRATION = True  # start from calibration/auto_<video>.json when it exists
    CAMERA_HFOV_DEG   = 65.0       # field of view across the image's LONG side, assumed by the
                                   # estimate (a portrait phone video sees ~39° across its width)
    PERSON_HEIGHT_M   = 1.676      # average full-body height (5'6"), the estimate's ruler
    CALIB_MIN_SPAN    = 0.15       # feet must spread over this share of the image height to
                                   # measure perspective; otherwise heads are used
    CALIB_MIN_BOXES   = 300        # full-body boxes needed before estimating
    CALIB_MIN_FRAMES  = 20
    CALIB_MAX_FRAMES  = 300        # give up and stay flat after this many frames
    MAX_DEPTH_RATIO   = 25.0       # ignore ground > 25× farther than the image bottom
    FLOOR_MIN_FEET    = 200        # feet seen before estimating where the floor ends
    SCENE_WIDTH_M     = 40.0       # flat fallback: ground covered by the view, in metres
    SCENE_HEIGHT_M    = 22.5

    # Zones
    ZONE_ROWS         = 2
    ZONE_COLS         = 3

    # Risk thresholds
    DENSITY_HIGH      = 0.40       # persons/m²; density risk saturates at 2× = 0.8 p/m² (Fruin LOS F)
    SPEED_HIGH        = 1.5        # m/s; speed risk saturates at 2× = 3 m/s (running)
    # Feet position noise of the detector, for the motion filter (forecast.MotionFilter)
    FEET_JITTER_PX    = 1.5        # px, at least (small boxes jitter by ~1–2 px whatever their size)
    FEET_JITTER_X     = 0.015      # × box height, across
    FEET_JITTER_Y     = 0.03       # × box height, along (the box bottom is fuzzier)
    RISK_WINDOW       = 15         # frames for trend window

    # Simulation (Layers J + K)
    PRED_HORIZON      = 25         # steps
    SIM_STEP_S        = 0.8        # seconds per step → 25 × 0.8 s = 20 s ahead
    SIM_SUBSTEPS      = 4          # integration substeps per step
    SIM_FUSION_CONF   = 0.70       # also trigger when fusion confidence κ drops below this
    TRACKER_BUFFER    = 30         # processed frames a lost track stays in the twin

    # 3D digital twin (/twin) — runs the 20 s forecast, separate from the dashboard
    TWIN_HZ           = 5          # crowd-state updates to the 3D twin per second
    FORECAST_HZ       = 1          # forecasts per second, while the twin is open or flagged
    FORECAST_ALERTS   = True       # when the trigger flags the scene, forecast even with no
                                   # viewer and alert on zones predicted to reach capacity
    FORECAST_ALERT_MAX_S = 20.0    # … within this many seconds
    OBSTACLE_EVERY_S  = 30.0       # how often the twin re-learns obstacles (never-walked floor)
    OBSTACLE_MIN_PERSON_PX = 32    # only where a person would be this tall in the image: farther
                                   # away, empty floor may just be people the detector misses
    WHATIF_STEPS      = 50         # panic what-if: 50 × SIM_STEP_S = 40 s

    # Social force model (Helbing & Molnár, 1995) — metres and seconds
    SF_TAU            = 0.5        # relaxation time towards the desired velocity
    SF_V0             = 2.1        # repulsive potential strength (m²/s²)
    SF_SIGMA          = 0.3        # repulsive potential range (m)
    SF_STEP           = 2.0        # look-ahead: the other pedestrian's next step (s)
    SF_FOV_DEG        = 200.0      # field of view; people outside it count less
    SF_FOV_WEIGHT     = 0.5        # weight of people outside the field of view
    SF_CUTOFF         = 3.0        # interactions beyond this distance are ignored (m)
    SF_MAX_SPEED      = 2.5        # m/s

    # Fusion weights
    W_VIDEO           = 0.70
    W_IOT             = 0.30
    DISCREP_THR       = 5          # persons before flagging discrepancy
    CONF_BOOST_MAX    = 1.2        # cap on the occlusion confidence boost

    # IoT gate counters (optional add-on; off = video-only fusion)
    IOT_ENABLED       = False      # --iot (or --mqtt) turns the gate stream on
    IOT_NOISE_STD     = 2.0        # std-dev of the gate miscount (persons)
    IOT_ERR_DECAY     = 0.9        # AR(1) decay of the miscount — keeps it bounded
    IOT_OCCLUSION     = 1.15       # gates also count people the camera cannot see
    IOT_SMOOTHING     = 0.1        # occupancy follows the video count at this EMA rate
    IOT_THROUGH_RATE  = 0.02       # share of the occupancy passing through the gates per tick
    IOT_TOKEN         = None       # set to require header X-IoT-Token on POST /api/iot

    # Density
    HEATMAP_SIGMA_M   = 1.0        # Gaussian per person, in metres on the ground

    # Output
    SEND_FRAME        = True       # False → heatmap without any video pixels in the payload
    OVERLAY_DETAIL    = "clean"    # dashboard video: "clean" (boxes, feet, zones) or "full" (+ ids, heads)
    WS_HZ             = 10         # max broadcasts per second
    EXPERIENCE_MAX_GB = 2.0        # oldest experience logs/frames are deleted beyond this

    # Videos the dashboard's source switcher offers
    VIDEO_DIR         = "videos"
    VIDEO_EXTS        = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    ALLOW_STREAM_URLS = True       # the switcher may also open cameras (index) and rtsp/http URLs


def resolve_device(pref: str) -> str:
    """'auto' → 'cuda:0' when PyTorch sees a GPU, else 'cpu'."""
    if pref != "auto":
        return pref
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def fit_within(h: int, w: int, max_h: int, max_w: int) -> Tuple[int, int]:
    """Largest (h, w) with the same aspect ratio that fits in max_h × max_w."""
    s = min(1.0, max_w / w, max_h / h)
    return max(1, int(round(h * s))), max(1, int(round(w * s)))


# ─── Ground Plane (perspective) ───────────────────────────────────────────────

def assumed_hfov(h: int, w: int, long_fov: Optional[float] = None) -> float:
    """Horizontal field of view for an h × w image whose long side sees long_fov."""
    long_fov = Config.CAMERA_HFOV_DEG if long_fov is None else long_fov
    if w >= h:
        return long_fov
    return math.degrees(2 * math.atan(math.tan(math.radians(long_fov) / 2) * w / h))


def person_height_px(v_foot, f: float, v0: float, pitch: float, cam_h: float,
                     person_h: float) -> np.ndarray:
    """
    Pixel height of a person whose feet are at image row v_foot, seen by a
    pinhole camera (focal length f px, centre row v0) cam_h metres above the
    ground and tilted down by `pitch` radians, with no roll.
    """
    v  = np.asarray(v_foot, float)
    t  = (v - v0) / f
    cp, sp = math.cos(pitch), math.sin(pitch)
    z  = cam_h * (cp - t * sp) / np.maximum(t * cp + sp, 1e-6)     # ground depth of the feet
    c2 = cam_h - person_h                                            # camera height above the head
    v_head = v0 + f * (c2 * cp - z * sp) / np.maximum(c2 * sp + z * cp, 1e-6)
    return v - v_head


class GroundPlane:
    """
    Maps image pixels to metres on the ground through a homography, so
    distances, areas and speeds account for perspective: in an oblique view a
    zone near the top of the frame covers far more ground than one at the
    bottom, and each pixel there spans more metres.

    Ground coordinates are metres (for camera models: X to the right, Y away
    from the camera). Image points at or near the horizon — more than
    MAX_DEPTH_RATIO times farther away than the bottom of the image — are out
    of range and get clipped or clamped.
    """
    def __init__(self, H: np.ndarray, h: int, w: int, source: str,
                 params: Optional[dict] = None):
        H = np.asarray(H, float)
        # Normalise so the homogeneous scale is +1 at the bottom-centre pixel
        w_bottom = float(H[2] @ [w / 2, h - 1, 1])
        if abs(w_bottom) < 1e-12:
            raise ValueError("The bottom of the image maps to the horizon.")
        self.H      = H / w_bottom
        self.H_inv  = np.linalg.inv(self.H)
        self.h, self.w = h, w
        self.source = source
        self.params = params or {}
        self.w_min  = 1.0 / Config.MAX_DEPTH_RATIO
        # Image row where the walkable floor ends on the far side (walls, sky above)
        self.floor_top: Optional[float] = None

    # ── Construction ──────────────────────────────────────────────────────
    @classmethod
    def flat(cls, h: int, w: int, width_m: Optional[float] = None,
             height_m: Optional[float] = None, source: str = "flat") -> "GroundPlane":
        """No perspective: the view covers width_m × height_m of ground evenly."""
        width_m  = width_m or Config.SCENE_WIDTH_M
        height_m = height_m or Config.SCENE_HEIGHT_M
        sx, sy = width_m / w, height_m / h
        # Like the camera models: X to the right, Y away from the camera (up the image)
        return cls(np.array([[sx, 0, 0], [0, -sy, h * sy], [0, 0, 1.0]]), h, w, source,
                   {"width_m": width_m, "height_m": height_m})

    @classmethod
    def from_camera(cls, h: int, w: int, height_m: float, pitch_deg: float,
                    hfov_deg: float, source: str = "camera") -> "GroundPlane":
        """Pinhole camera height_m above flat ground, tilted down pitch_deg, no roll."""
        f  = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
        u0, v0 = w / 2, h / 2
        c  = height_m
        cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
        # Ground (X right, Y away from the camera's foot) → image, homogeneous
        world_to_image = np.array([
            [f, u0 * cp,          u0 * c * sp],
            [0, v0 * cp - f * sp, v0 * c * sp + f * c * cp],
            [0, cp,               c * sp],
        ])
        return cls(np.linalg.inv(world_to_image), h, w, source,
                   {"camera_height_m": height_m, "pitch_deg": pitch_deg, "hfov_deg": hfov_deg})

    @classmethod
    def from_points(cls, h: int, w: int, image_pts, world_pts,
                    source: str = "points") -> "GroundPlane":
        """Homography from ≥4 image points and their positions on the ground (metres)."""
        img, gnd = np.float32(image_pts), np.float32(world_pts)
        if len(img) < 4 or len(img) != len(gnd):
            raise ValueError("Need at least 4 matching image and ground points.")
        H, _ = cv2.findHomography(img, gnd, 0)
        if H is None:
            raise ValueError("The points don't define a ground plane (are 3 of them in a line?).")
        return cls(H, h, w, source, {"points": len(img)})

    @classmethod
    def from_file(cls, path, h: int, w: int) -> "GroundPlane":
        """
        Load a calibration JSON in one of three forms:
          {"image_points": [[x, y], ...], "world_points": [[X, Y], ...],
           "image_size": [W, H]}                  ≥4 ground points; pixels of image_size
          {"camera_height_m": 6.5, "pitch_deg": 30, "hfov_deg": 70}
          {"scene_width_m": 40, "scene_height_m": 22.5}               flat
        """
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if "image_points" in d:
            pts  = np.asarray(d["image_points"], float)
            size = d.get("image_size")
            if size:                                # pixels of another resolution
                pts = pts * [w / size[0], h / size[1]]
            elif pts.max() <= 1.0:                  # fractions of the frame
                pts = pts * [w, h]
            return cls.from_points(h, w, pts, d["world_points"], source="calibration")
        if "camera_height_m" in d:
            return cls.from_camera(h, w, d["camera_height_m"], d["pitch_deg"],
                                   d.get("hfov_deg", assumed_hfov(h, w)), source="calibration")
        if "scene_width_m" in d:
            return cls.flat(h, w, d["scene_width_m"], d["scene_height_m"], source="calibration")
        raise ValueError(f"{path}: expected image_points + world_points, camera_height_m, "
                         f"or scene_width_m")

    def save_dict(self) -> dict:
        """This calibration in the file format from_file reads."""
        p = self.params
        if "camera_height_m" in p:
            d = {k: round(float(p[k]), 3) for k in ("camera_height_m", "pitch_deg", "hfov_deg")}
        elif "width_m" in p:
            d = {"scene_width_m": round(p["width_m"], 3), "scene_height_m": round(p["height_m"], 3)}
        else:
            raise ValueError("Point calibrations are saved by calibrate.py.")
        return {**d, "image_size": [self.w, self.h], "source": self.source}

    def describe(self) -> dict:
        """Summary for the dashboard."""
        d = {"source": self.source,
             **{k: round(v, 2) if isinstance(v, float) else v for k, v in self.params.items()}}
        if self.floor_top is not None:
            d["floor_top_y"] = int(self.floor_top)
        return d

    # ── Mapping ───────────────────────────────────────────────────────────
    def _scale(self, pts: np.ndarray) -> np.ndarray:
        """Homogeneous scale of image points: 1 at the bottom, 0 on the horizon."""
        return pts @ self.H[2, :2] + self.H[2, 2]

    def to_world(self, pts) -> np.ndarray:
        """(N, 2) image pixels → (N, 2) metres on the ground; out of range is clamped."""
        p   = np.asarray(pts, float).reshape(-1, 2)
        hom = p @ self.H[:, :2].T + self.H[:, 2]
        return hom[:, :2] / np.maximum(hom[:, 2], self.w_min)[:, None]

    def to_image(self, pts) -> np.ndarray:
        """(N, 2) metres on the ground → (N, 2) image pixels."""
        p   = np.asarray(pts, float).reshape(-1, 2)
        hom = p @ self.H_inv[:, :2].T + self.H_inv[:, 2]
        return hom[:, :2] / hom[:, 2:3]

    def person_px(self, pts) -> np.ndarray:
        """Roughly how tall (px) a PERSON_HEIGHT_M person standing at image points
        appears: their height over the local sideways ground scale."""
        p = np.asarray(pts, float).reshape(-1, 2)
        a, b = self.to_world(p - [0.5, 0]), self.to_world(p + [0.5, 0])
        return Config.PERSON_HEIGHT_M / np.maximum(np.linalg.norm(b - a, axis=1), 1e-6)

    def m_per_px(self, pts) -> np.ndarray:
        """Local ground scale (metres per pixel) at image points."""
        s = np.maximum(self._scale(np.asarray(pts, float).reshape(-1, 2)), self.w_min)
        return np.sqrt(abs(np.linalg.det(self.H)) / s ** 3)

    @staticmethod
    def _clip_half(pts: np.ndarray, value) -> np.ndarray:
        """Sutherland–Hodgman: the part of a polygon where the affine value(p) ≥ 0."""
        if len(pts) == 0:
            return pts
        s   = value(pts)
        out = []
        for k in range(len(pts)):
            a, b, sa, sb = pts[k - 1], pts[k], s[k - 1], s[k]
            if sb >= 0:
                if sa < 0:
                    out.append(a + (b - a) * sa / (sa - sb))
                out.append(b)
            elif sa >= 0:
                out.append(a + (b - a) * sa / (sa - sb))
        return np.array(out, float).reshape(-1, 2)

    def clip(self, poly) -> np.ndarray:
        """The part of an image polygon that shows walkable ground in range."""
        pts = np.asarray(poly, float).reshape(-1, 2)
        pts = self._clip_half(pts, lambda p: self._scale(p) - self.w_min)    # past the horizon
        if self.floor_top is not None:
            pts = self._clip_half(pts, lambda p: p[:, 1] - self.floor_top)   # walls / sky
        return pts

    def area_m2(self, poly) -> float:
        """Ground area (m²) covered by an image polygon."""
        g = self.to_world(self.clip(poly))
        if len(g) < 3:
            return 0.0
        x, y = g[:, 0], g[:, 1]
        return float(abs(x @ np.roll(y, -1) - y @ np.roll(x, -1)) / 2)

    def visible_ground(self) -> np.ndarray:
        """The in-range ground in view, as a convex polygon in metres."""
        return self.to_world(self.clip([[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]]))

    def boundary(self) -> List[dict]:
        """
        Edges of the visible floor in metres, each tagged with where it comes
        from: "wall" — the floor ends there (walls, sky; learned from people's
        feet), "view" — the edge of the camera's view, or "range" — too far
        away to use.
        """
        img = self.clip([[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]])
        if len(img) < 3:
            return []
        world  = self.to_world(img)
        on_far = np.abs(self._scale(img) - self.w_min) < 1e-9
        edges  = []
        for k in range(len(img)):
            a, b = img[k - 1], img[k]
            if (self.floor_top is not None and abs(a[1] - self.floor_top) < 1e-6
                    and abs(b[1] - self.floor_top) < 1e-6):
                kind = "wall"
            elif on_far[k - 1] and on_far[k]:
                kind = "range"
            else:
                kind = "view"
            edges.append({"a": np.round(world[k - 1], 2).tolist(),
                          "b": np.round(world[k], 2).tolist(), "kind": kind})
        return edges

    def feet_from_heads(self, boxes) -> Tuple[np.ndarray, np.ndarray]:
        """
        Where people stand, from head boxes (x1, y1, x2, y2) → feet pixels
        (N, 2) and body heights in px (N,). With a camera model this is exact
        geometry: the chin lies on a plane PERSON_HEIGHT_M − HEAD_SIZE_M above
        the floor, straight above the feet, so the chin's point on that plane
        is the feet's point on the floor. Otherwise a person is BODY_PER_HEAD
        heads tall.
        """
        b  = np.asarray(boxes, float).reshape(-1, 4)
        cx = (b[:, 0] + b[:, 2]) / 2
        hh = np.maximum(b[:, 3] - b[:, 1], 1.0)
        feet = np.column_stack([cx, b[:, 3] + (Config.BODY_PER_HEAD - 1) * hh])
        p = self.params
        lift = Config.PERSON_HEIGHT_M - Config.HEAD_SIZE_M
        if len(b) and "camera_height_m" in p and p["camera_height_m"] > lift + 0.3:
            chin = GroundPlane.from_camera(self.h, self.w, p["camera_height_m"] - lift,
                                           p["pitch_deg"], p["hfov_deg"])
            f = self.to_image(chin.to_world(np.column_stack([cx, b[:, 3]])))
            ok = np.isfinite(f).all(axis=1) & (f[:, 1] > b[:, 3])
            feet[ok] = f[ok]
        return feet, feet[:, 1] - b[:, 1]

    def camera(self) -> Optional[dict]:
        """
        Camera pose for the 3D twin: height, tilt and field of view, and where
        on the ground it stands (x, y) and which way it looks (yaw from +Y
        towards +X). Camera models stand at the origin looking along +Y; for
        point calibrations the pose is recovered from the homography.
        """
        p = self.params
        if "camera_height_m" in p:
            return {"height_m": round(float(p["camera_height_m"]), 2),
                    "pitch_deg": round(float(p["pitch_deg"]), 2),
                    "hfov_deg": round(float(p["hfov_deg"]), 2),
                    "x": 0.0, "y": 0.0, "yaw_deg": 0.0}
        return self.pose_from_homography()

    def pose_from_homography(self) -> Optional[dict]:
        """
        Decompose ground → image = K [r1 r2 t] for a pinhole camera with square
        pixels and the principal point at the image centre: the focal length
        makes r1, r2 orthonormal (least squares over both conditions), then
        the camera's centre is −Rᵀt. None when the view has no perspective
        (flat calibrations) or the homography isn't a real camera's.
        """
        G = self.H_inv / np.linalg.norm(self.H_inv)                   # ground → image
        T = np.array([[1, 0, -self.w / 2], [0, 1, -self.h / 2], [0, 0, 1.0]])
        M = T @ G
        m1, m2, m3 = M[:, 0], M[:, 1], M[:, 2]
        a = np.array([m1[0] * m2[0] + m1[1] * m2[1], (m1[:2] @ m1[:2]) - (m2[:2] @ m2[:2])])
        b = np.array([m1[2] * m2[2], m1[2] ** 2 - m2[2] ** 2])
        if (a @ a) < 1e-18:
            return None
        q = -(a @ b) / (a @ a)                                          # 1 / f²
        if not q > 0:
            return None
        f = 1.0 / math.sqrt(q)
        s = np.array([1 / f, 1 / f, 1.0])
        r1, r2, t = m1 * s, m2 * s, m3 * s
        lam = 2.0 / (np.linalg.norm(r1) + np.linalg.norm(r2))
        if t[2] * lam < 0:                                              # ground in front of the camera
            lam = -lam
        r1, r2, t = r1 * lam, r2 * lam, t * lam
        U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
        R = U @ Vt                                                      # nearest rotation
        C = -R.T @ t                                                    # camera centre, ground frame
        up = 1.0 if C[2] >= 0 else -1.0                                 # handedness of the user's axes
        d = R[2, :]                                                     # optical axis, ground frame
        pitch = math.degrees(math.asin(np.clip(-d[2] * up, -1, 1)))
        height = abs(float(C[2]))
        if height < 0.2 or not 0 < pitch < 90:
            return None
        return {"height_m": round(height, 2), "pitch_deg": round(pitch, 2),
                "hfov_deg": round(math.degrees(2 * math.atan(self.w / 2 / f)), 2),
                "x": round(float(C[0]), 2), "y": round(float(C[1]), 2),
                "yaw_deg": round(math.degrees(math.atan2(d[0], d[1])), 2)}


class PedestrianCalibrator:
    """
    Estimates the ground plane from people's full-body boxes (single-view
    metrology). In a perspective view a person's height in pixels grows
    linearly with how far below the horizon their feet are: the horizon row
    and the slope give the camera's tilt and height, for an assumed field of
    view (CAMERA_HFOV_DEG) and person height (PERSON_HEIGHT_M). The FBOX
    model predicts whole-body boxes even for partly hidden people, so boxes
    inside a crowd still measure whole people. In a dense crowd, where
    bodies are hidden, heads do the same job: they sit on a plane
    PERSON_HEIGHT_M − HEAD_SIZE_M above the floor and measure HEAD_SIZE_M.
    """
    def __init__(self, h: int, w: int):
        self.h, self.w = h, w
        self.frames    = 0
        self._feet: List[float]    = []
        self._heights: List[float] = []
        self._chins: List[float]   = []          # head boxes: bottom row …
        self._heads: List[float]   = []          # … and height, px

    def add(self, dets: List[dict], heads: Sequence[dict] = ()):
        self.frames += 1
        for hd in heads:
            x1, y1, x2, y2 = hd["box"]
            if hd.get("confidence", 1.0) >= 0.3 and y1 > 2 and y2 < self.h - 2 and \
                    0.6 <= (y2 - y1) / max(x2 - x1, 1e-6) <= 2.0:
                self._chins.append(float(y2))
                self._heads.append(float(y2 - y1))
        for d in dets:
            fy, bh = d.get("fy"), d.get("bh")
            if not bh or d.get("confidence", 1.0) < 0.4 or d.get("src", "body") != "body":
                continue
            if fy - bh <= 2 or fy >= self.h - 2:             # cut off at the top or bottom
                continue
            if "x1" in d:
                bw = d["x2"] - d["x1"]
                if d["x1"] <= 2 or d["x2"] >= self.w - 2 or not 1.5 <= bh / max(bw, 1e-6) <= 5.0:
                    continue                                  # cut off at a side, or not upright
            self._feet.append(float(fy))
            self._heights.append(float(bh))

    @property
    def ready(self) -> bool:
        return ((self.frames >= Config.CALIB_MIN_FRAMES and
                 (len(self._feet) >= Config.CALIB_MIN_BOXES or
                  len(self._heads) >= Config.CALIB_MIN_HEADS)) or
                self.frames >= Config.CALIB_MAX_FRAMES)

    def fit(self) -> Tuple[Optional[GroundPlane], str]:
        """The estimated ground plane (None if it can't be estimated) and why."""
        # Perspective is measured across rows: a ruler is only as good as the
        # spread of the rows it was seen at. Whole bodies are the better ruler
        # (heads all sit near the horizon when the camera is at head height),
        # so they are used when their feet cover CALIB_MIN_SPAN of the image;
        # otherwise heads (a dense crowd, or only the front row fully visible).
        span = lambda r: (float(np.percentile(r, 95) - np.percentile(r, 5)) / self.h
                          if len(r) else 0.0)
        bodies = (lambda: self._fit(np.array(self._feet), np.array(self._heights),
                                    Config.PERSON_HEIGHT_M, 0.0, "people"))             if len(self._feet) >= 30 else None
        heads = (lambda: self._fit(np.array(self._chins), np.array(self._heads),
                                   Config.HEAD_SIZE_M,
                                   Config.PERSON_HEIGHT_M - Config.HEAD_SIZE_M, "heads"))             if len(self._heads) >= 60 else None
        sb, sh = span(self._feet), span(self._chins)
        if bodies and sb >= Config.CALIB_MIN_SPAN:
            fits = [bodies, heads]
        elif heads and (sh >= Config.CALIB_MIN_SPAN or sh > sb):
            fits = [heads, bodies]
        else:
            fits = [bodies, heads]
        fits = [f for f in fits if f is not None]
        if not fits:
            return None, f"only {len(self._feet)} usable full-body boxes"
        why_all = []
        for fit in fits:
            ground, why = fit()
            if ground is not None:
                return ground, why
            why_all.append(why)
        return None, "; ".join(why_all)

    def _fit(self, y: np.ndarray, hp: np.ndarray, size_m: float, lift_m: float,
             what: str) -> Tuple[Optional[GroundPlane], str]:
        """
        Fit the camera from objects of real height size_m whose bottom rows
        are y and pixel heights hp, standing on a plane lift_m above the floor
        (0 for whole people; the chin plane for heads).
        """
        n_all = len(y)
        # Robust line  height = a · bottom_row + b  (RANSAC, then least squares)
        rng, best = np.random.default_rng(0), None
        for _ in range(300):
            i, j = rng.choice(len(y), 2, replace=False)
            if abs(y[i] - y[j]) < 0.05 * self.h:
                continue
            a   = (hp[i] - hp[j]) / (y[i] - y[j])
            inl = np.abs(hp - (hp[i] + a * (y - y[i]))) < np.maximum(0.15 * hp, 3.0)
            if best is None or inl.sum() > best.sum():
                best = inl
        if best is None or best.sum() < max(30, 0.3 * n_all):
            return None, "box heights don't follow a single ground plane"
        y, hp = y[best], hp[best]
        a, b = np.polyfit(y, hp, 1)
        # Little perspective (distant or overhead view): the sizes would only
        # shrink to nothing at a horizon far above the frame. Judged by the
        # horizon, not by the size change across the rows seen — a narrow band
        # of rows hardly changes even under strong perspective.
        if a <= 0:
            # Sizes don't grow towards the camera: in a packed crowd the detector's
            # near boxes are cut and far ones inflated — no ground plane in them
            return None, f"the {what} don't look bigger nearer the camera"
        if -b / a < -5.0 * self.h:
            # Hardly any perspective (distant or overhead view): one scale from their size
            s = size_m / float(np.median(hp))
            ground = GroundPlane.flat(self.h, self.w, s * self.w, s * self.h, source="pedestrians")
            return ground, f"little perspective; {s * 100:.1f} cm per pixel from {len(y)} {what}"
        horizon = -b / a
        if horizon > y.min() - 5:
            return None, f"the estimated horizon falls below some {what}"
        # Exact pinhole model with the assumed field of view: fit tilt and height
        hfov = assumed_hfov(self.h, self.w)
        f  = (self.w / 2) / math.tan(math.radians(hfov) / 2)
        v0 = self.h / 2
        lo, hi = [-0.3, 0.3], [1.5, 500.0]
        x0 = np.clip([math.atan2(v0 - horizon, f), size_m / a], lo, hi)
        fit = least_squares(
            lambda p: person_height_px(y, f, v0, p[0], p[1], size_m) / hp - 1,
            x0=x0, bounds=(lo, hi), loss="soft_l1", f_scale=0.1,
        )
        pitch, cam_h = fit.x
        cam_h = float(cam_h) + lift_m                       # above the floor
        ground = GroundPlane.from_camera(self.h, self.w, cam_h, math.degrees(pitch),
                                         hfov, source="pedestrians")
        return ground, (f"camera ≈ {cam_h:.1f} m high, tilted {math.degrees(pitch):.0f}° down "
                        f"(from {len(y)} {what}, {hfov:.0f}° horizontal field of view assumed)")


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class Agent:
    id:         int
    x:          float            # feet in the image, px
    y:          float
    wx:         float = 0.0      # feet on the ground, metres
    wy:         float = 0.0
    vx:         float = 0.0      # ground velocity, m/s
    vy:         float = 0.0
    speed:      float = 0.0      # m/s
    vel_std:    float = 0.0      # uncertainty of the velocity, m/s
    walking:    float = 0.0      # 0 = standing, 1 = walking (speed clearly above its noise)
    confidence: float = 1.0
    zone:       str   = ""
    last_seen:  float = 0.0      # twin clock (s) at the last detection
    missed:     int   = 0        # consecutive processed frames without a detection


@dataclass
class ZoneState:
    name:     str
    x1:       int                      # bounding box of the zone in the image (px)
    y1:       int
    x2:       int
    y2:       int
    capacity: int   = 50
    area_m2:  float = 0.0
    polygon:  Optional[np.ndarray] = None   # the zone on the floor (m), convex
    image_polygon: Optional[np.ndarray] = None   # the same, in the image (px)

    @property
    def area(self) -> float:
        return float((self.x2 - self.x1) * (self.y2 - self.y1))

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2


def _poly_area(p: np.ndarray) -> float:
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _equal_area_cuts(poly: np.ndarray, axis: int, n: int) -> List[float]:
    """Values v_1 < … < v_(n−1) of coordinate `axis` that cut a convex polygon
    into n slices of equal area."""
    lo, hi = float(poly[:, axis].min()), float(poly[:, axis].max())
    total, cuts = _poly_area(poly), []
    for k in range(1, n):
        target, a, b = total * k / n, lo, hi
        for _ in range(50):                                  # bisection on the slice area
            m = (a + b) / 2
            part = GroundPlane._clip_half(poly, lambda p, m=m: m - p[:, axis])
            a, b = (m, b) if _poly_area(part) < target else (a, m)
        cuts.append((a + b) / 2)
    return cuts


def _slab(poly: np.ndarray, axis: int, lo: Optional[float], hi: Optional[float]) -> np.ndarray:
    if lo is not None:
        poly = GroundPlane._clip_half(poly, lambda p: p[:, axis] - lo)
    if hi is not None:
        poly = GroundPlane._clip_half(poly, lambda p: hi - p[:, axis])
    return poly


# ─── Zone Manager ─────────────────────────────────────────────────────────────

class ZoneManager:
    """
    Zones on the floor, not on the image: the walkable floor in view (clipped at
    the learned far edge) is cut into ROWS depth bands of equal floor area, and
    each band into COLS parts of equal area, so every zone covers the same
    number of square metres however the camera looks at it. Zone_A … Zone_C are
    the far band (top of the image), Zone_D … Zone_F the near one. People are
    assigned by where they stand, in metres.
    """
    def __init__(self, h: int, w: int, ground: GroundPlane,
                 rows: int = Config.ZONE_ROWS, cols: int = Config.ZONE_COLS):
        self.zones: Dict[str, ZoneState] = {}
        self.ground = ground
        self.h, self.w = h, w
        self.rows, self.cols = rows, cols
        floor = ground.visible_ground()
        bands = []
        if _poly_area(floor) > 1e-6:
            ycut = _equal_area_cuts(floor, 1, rows)[::-1]     # far band first
            edges = [None] + ycut + [None]
            for r in range(rows):
                hi, lo = edges[r], edges[r + 1]               # far → near: y from high to low
                bands.append(_slab(floor, 1, lo, hi))
        for r in range(rows):
            band = bands[r] if bands else np.zeros((0, 2))
            xcut = _equal_area_cuts(band, 0, cols) if _poly_area(band) > 1e-6 else []
            xe = [None] + xcut + [None]
            for c in range(cols):
                name = self.names(rows, cols)[r * cols + c]
                poly = _slab(band, 0, xe[c], xe[c + 1]) if xcut or cols == 1 else np.zeros((0, 2))
                img = ground.to_image(poly) if len(poly) else np.zeros((0, 2))
                if len(img):
                    x1, y1 = np.floor(img.min(0)).astype(int)
                    x2, y2 = np.ceil(img.max(0)).astype(int)
                else:
                    x1 = y1 = x2 = y2 = 0
                z = ZoneState(name=name, x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                              polygon=poly, image_polygon=img)
                z.area_m2  = _poly_area(poly)
                # Capacity = persons at the Fruin LOS F density (2 × DENSITY_HIGH)
                z.capacity = max(10, int(z.area_m2 * Config.DENSITY_HIGH * 2))
                self.zones[name] = z
        self._names = list(self.zones)
        # Half-plane form of each convex zone, for vectorised assignment
        self._half = []
        for z in self.zones.values():
            p = z.polygon if z.polygon is not None and len(z.polygon) >= 3 else None
            if p is None:
                self._half.append(None)
                continue
            if np.cross(p[1] - p[0], p[2] - p[1]) < 0:        # make it counter-clockwise
                p = p[::-1]
            e = np.roll(p, -1, axis=0) - p
            n = np.stack([-e[:, 1], e[:, 0]], 1)              # inward normals (CCW)
            self._half.append((n, np.einsum("ij,ij->i", n, p)))
        cent = [z.polygon.mean(0) if z.polygon is not None and len(z.polygon) else np.full(2, np.inf)
                for z in self.zones.values()]
        self._centres = np.array(cent, float).reshape(-1, 2)

    @staticmethod
    def names(rows: int = Config.ZONE_ROWS, cols: int = Config.ZONE_COLS) -> List[str]:
        return [f"Zone_{'ABCDEFGHIJKLMNOP'[i]}" for i in range(rows * cols)]

    def index_world(self, pts) -> np.ndarray:
        """Zone index (in names() order) for (N, 2) ground points in metres;
        points off the floor go to the nearest zone."""
        p = np.asarray(pts, float).reshape(-1, 2)
        out = np.full(len(p), -1, int)
        for i, hp in enumerate(self._half):
            if hp is None:
                continue
            n, off = hp
            inside = np.all(p @ n.T >= off - 1e-6, axis=1) & (out < 0)
            out[inside] = i
        miss = out < 0
        if miss.any() and np.isfinite(self._centres).any():
            d = np.linalg.norm(p[miss, None, :] - self._centres[None], axis=2)
            out[miss] = np.argmin(d, axis=1)
        return np.maximum(out, 0)

    def index_many(self, pts) -> np.ndarray:
        """Zone index (in names() order) for (N, 2) image points."""
        p = np.asarray(pts, float).reshape(-1, 2)
        if not len(p):
            return np.zeros(0, int)
        return self.index_world(self.ground.to_world(p))

    def assign(self, cx: float, cy: float) -> str:
        return self._names[int(self.index_many([[cx, cy]])[0])]

    def polygons(self) -> Dict[str, np.ndarray]:
        """Each zone's walkable ground in view, as a polygon in metres."""
        return {name: (z.polygon if z.polygon is not None else np.zeros((0, 2)))
                for name, z in self.zones.items()}

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
        """Per-zone confidence weight (manual calibration; 1.0 = neutral)."""
        self._zone_cal[zone] = max(0.5, min(1.0, weight))

    def drop(self, agent_id: int):
        self._age.pop(agent_id, None)


# ─── IoT Simulator ────────────────────────────────────────────────────────────

class IoTSimulator:
    """
    Entry/exit gate counters — fusion Stream 2.

    Simulation mode (default): the gates track the venue's real occupancy,
    modelled as the video count smoothed over time (detections flicker,
    people don't) and scaled up for people the camera cannot see. People
    also pass through the gates in both directions. Miscounts follow a
    mean-reverting AR(1) process, so the net count stays within a few
    persons of the occupancy instead of drifting away over time.

    Live mode: call push_real(entry, exit_) from the sensor callback and the
    simulation switches off, e.g. with MQTT:
        import paho.mqtt.client as mqtt
        client.subscribe("venue/gate/entry")
        def on_message(client, userdata, msg): iot.push_real(int(msg.payload), 0)
    """
    def __init__(self, noise_std: float = Config.IOT_NOISE_STD):
        self.noise_std          = noise_std
        self.live               = False
        self._cumulative_entry  = 0
        self._cumulative_exit   = 0
        self._level: Optional[float] = None    # smoothed occupancy seen by the gates
        self._occupancy         = 0
        self._err               = 0.0
        self._pending           = [0, 0]       # live counts since the last tick
        self._lock              = threading.Lock()
        self._history           = deque(maxlen=200)

    @property
    def mode(self) -> str:
        if self.live:
            return "live"
        return "simulated" if Config.IOT_ENABLED else "off"

    def tick(self, video_count: int, frame_idx: int) -> Tuple[int, int]:
        """Advance one pipeline frame; returns the (entry, exit) counts since the last tick."""
        if self.live:
            with self._lock:
                entry, exit_  = self._pending
                self._pending = [0, 0]
        else:
            entry, exit_ = self._simulate(video_count)
        self._history.append((frame_idx, entry, exit_))
        return entry, exit_

    def _simulate(self, video_count: int) -> Tuple[int, int]:
        seen = video_count * Config.IOT_OCCLUSION
        if self._level is None:
            # The gates were already counting when the camera feed started.
            self._level            = seen
            self._occupancy        = int(round(seen))
            self._cumulative_entry = self._occupancy
        self._level += Config.IOT_SMOOTHING * (seen - self._level)
        target  = int(round(self._level))
        through = int(np.random.poisson(Config.IOT_THROUGH_RATE * max(target, 1)))
        entry   = max(0, target - self._occupancy) + through
        exit_   = max(0, self._occupancy - target) + through
        self._occupancy         = target
        self._cumulative_entry += entry
        self._cumulative_exit  += exit_
        d = Config.IOT_ERR_DECAY
        self._err = d * self._err + np.random.normal(0, self.noise_std * math.sqrt(1 - d * d))
        return entry, exit_

    def push_real(self, entry: int, exit_: int):
        """Feed real gate counts (sensor callback); switches off the simulation."""
        with self._lock:
            if not self.live:
                self.live = True
                self._cumulative_entry = self._cumulative_exit = 0
                self._err = 0.0
            self._cumulative_entry += int(entry)
            self._cumulative_exit  += int(exit_)
            self._pending[0]       += int(entry)
            self._pending[1]       += int(exit_)

    @property
    def net_count(self) -> int:
        return max(0, int(round(self._cumulative_entry - self._cumulative_exit + self._err)))


# ─── Data Fusion Layer ────────────────────────────────────────────────────────

class DataFusionLayer:
    """
    Multi-modal sensor fusion:
      Stream 1: Video detections (YOLO26s + ByteTrack)    weight=0.70
      Stream 2: IoT gate counters (entry/exit cumulative) weight=0.30

    Key insight: video undercounts in crowds (occlusion).
    IoT overcounts slightly (re-entry, sensor noise).
    Weighted fusion + discrepancy detection corrects both.
    """

    def __init__(self):
        self.iot             = IoTSimulator()
        self._log            = deque(maxlen=300)

    def fuse(self, video_detections: list, frame_idx: int) -> dict:
        """
        Call every frame. Returns fusion result dict consumed by DT engine.

        Paper equation:
          C_f = W_VIDEO × C_video + W_IOT × C_iot
          κ   = 1 − (|C_video − C_iot| / max(C_video, C_iot, 1))
        """
        video_count = len(video_detections)

        # IoT switched off: video-only fusion, nothing to disagree with
        if not (Config.IOT_ENABLED or self.iot.live):
            result = {
                "fused_detections":  [dict(d) for d in video_detections],
                "fused_count":       float(video_count),
                "video_count":       video_count,
                "iot_count":         None,
                "iot_entry_delta":   None,
                "iot_exit_delta":    None,
                "discrepancy":       0,
                "discrepancy_flag":  False,
                "conf_scale":        1.0,
                "fusion_confidence": 1.0,
            }
            self._log.append(result)
            return result

        # Stream 2: IoT tick
        entry, exit_ = self.iot.tick(video_count, frame_idx)
        iot_count    = self.iot.net_count

        # Discrepancy detection
        discrepancy      = abs(video_count - iot_count)
        discrepancy_flag = discrepancy > Config.DISCREP_THR

        # Confidence rescaling direction
        if discrepancy_flag:
            if iot_count > video_count:
                # IoT sees more → occlusion → boost conf
                conf_scale = min(Config.CONF_BOOST_MAX, 1.0 + discrepancy * 0.02)
            else:
                # Video sees more → false dets → lower conf
                conf_scale = max(0.6, 1.0 - discrepancy * 0.02)
        else:
            conf_scale = 1.0

        # Rescale detection confidences by what the gates say
        fused_dets = []
        for det in video_detections:
            new_det = dict(det)
            new_det["confidence"] = round(min(1.0, det["confidence"] * conf_scale), 3)
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


# ─── Social Force Model ───────────────────────────────────────────────────────

class SocialForceModel:
    """
    Social force pedestrian model (Helbing & Molnár, 1995) — the twin's
    first forecast model, now kept as a baseline in evaluate.py (the 3D twin
    uses forecast.CrowdForecaster).

    Each pedestrian relaxes towards a desired velocity — the heading and
    speed it was tracked at — and is repelled by the others through the
    potential V(b) = V0·exp(−b/σ), where b is the semi-minor axis of an
    ellipse stretched along the other pedestrian's next step, so people
    start avoiding each other before they meet. People outside the field
    of view count less. Units are metres and seconds. With `bounds` — the
    visible ground as a convex polygon, or a (width, height) box — the area
    is closed and pedestrians reflect at its edges.
    """
    def __init__(self, tau: float = Config.SF_TAU, v0: float = Config.SF_V0,
                 sigma: float = Config.SF_SIGMA, step: float = Config.SF_STEP,
                 fov_deg: float = Config.SF_FOV_DEG,
                 fov_weight: float = Config.SF_FOV_WEIGHT,
                 cutoff: float = Config.SF_CUTOFF,
                 max_speed: float = Config.SF_MAX_SPEED):
        self.tau, self.v0, self.sigma = tau, v0, sigma
        self.step                     = step
        self.cos_fov                  = math.cos(math.radians(fov_deg / 2))
        self.fov_weight               = fov_weight
        self.cutoff                   = cutoff
        self.max_speed                = max_speed

    def simulate(self, pos: np.ndarray, vel: np.ndarray, steps: int, step_s: float,
                 substeps: int = Config.SIM_SUBSTEPS, bounds=None) -> List[np.ndarray]:
        """Advance (N, 2) positions and velocities; returns the positions after each step."""
        p = np.array(pos, dtype=float).reshape(-1, 2)
        v = np.array(vel, dtype=float).reshape(-1, 2)
        speed = np.linalg.norm(v, axis=1)
        v0 = np.minimum(speed, self.max_speed)                           # desired speed
        e0 = np.divide(v, speed[:, None], out=np.zeros_like(v),
                       where=speed[:, None] > 1e-9)                      # desired direction
        edges = self._edges(bounds) if bounds is not None else None
        dt  = step_s / substeps
        out = []
        for _ in range(steps):
            for _ in range(substeps):
                f = (v0[:, None] * e0 - v) / self.tau + self._repulsion(p, v)
                v = v + f * dt
                spd  = np.linalg.norm(v, axis=1)
                fast = spd > self.max_speed
                v[fast] *= (self.max_speed / spd[fast])[:, None]
                p = p + v * dt
                if edges is not None:
                    self._reflect(p, v, e0, *edges)
            out.append(p.copy())
        return out

    @staticmethod
    def _edges(bounds) -> Tuple[np.ndarray, np.ndarray]:
        """Inward unit normals and offsets of a convex polygon or a (width, height) box."""
        b = np.asarray(bounds, float)
        if b.ndim == 1:
            b = np.array([[0, 0], [b[0], 0], [b[0], b[1]], [0, b[1]]])
        centre = b.mean(axis=0)
        normals, offsets = [], []
        for a, c in zip(b, np.roll(b, -1, axis=0)):
            n = np.array([c[1] - a[1], a[0] - c[0]])
            norm = np.linalg.norm(n)
            if norm < 1e-9:
                continue
            n = n / norm
            if (centre - a) @ n < 0:
                n = -n
            normals.append(n)
            offsets.append(n @ a)
        return np.array(normals), np.array(offsets)

    @staticmethod
    def _reflect(p: np.ndarray, v: np.ndarray, e0: np.ndarray,
                 normals: np.ndarray, offsets: np.ndarray):
        """Bounce pedestrians that crossed an edge back inside (in place)."""
        for n, off in zip(normals, offsets):
            d   = p @ n - off
            out = d < 0
            if out.any():
                p[out]  -= 2 * d[out, None] * n
                v[out]  -= 2 * (v[out] @ n)[:, None] * n
                e0[out] -= 2 * (e0[out] @ n)[:, None] * n    # keep walking away from the edge
        # A bounce off one edge can overshoot another near a corner
        for n, off in zip(normals, offsets):
            d   = p @ n - off
            out = d < 0
            p[out] -= d[out, None] * n

    def _repulsion(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Force −∇V(b) on each pedestrian a from every pedestrian b nearby."""
        n = len(p)
        f = np.zeros_like(p)
        if n < 2:
            return f
        pairs = cKDTree(p).query_pairs(self.cutoff, output_type="ndarray")
        if len(pairs) == 0:
            return f
        ia = np.concatenate([pairs[:, 0], pairs[:, 1]])    # pedestrian feeling the force
        ib = np.concatenate([pairs[:, 1], pairs[:, 0]])    # pedestrian causing it
        r  = p[ia] - p[ib]
        s  = v[ib] * self.step                             # b's next step
        rn = np.maximum(np.linalg.norm(r, axis=1), 1e-9)
        rs = np.maximum(np.linalg.norm(r - s, axis=1), 1e-9)
        q  = rn + rs
        # 2b = sqrt((|r| + |r − s|)² − |s|²)
        b  = np.maximum(0.5 * np.sqrt(np.maximum(q ** 2 - (s ** 2).sum(axis=1), 0.0)), 1e-3)
        mag   = self.v0 / self.sigma * np.exp(-b / self.sigma) * q / (4 * b)
        force = mag[:, None] * (r / rn[:, None] + (r - s) / rs[:, None])
        # Field of view: someone behind a counts only fov_weight
        spd = np.linalg.norm(v[ia], axis=1)
        e   = np.divide(v[ia], spd[:, None], out=np.zeros_like(r), where=spd[:, None] > 1e-9)
        cos_phi = -np.einsum("ij,ij->i", e, r) / rn
        w = np.where((spd < 1e-9) | (cos_phi >= self.cos_fov), 1.0, self.fov_weight)
        force *= w[:, None]
        for k in range(2):
            f[:, k] = np.bincount(ia, force[:, k], minlength=n)
        return f


# ─── Digital Twin Engine ──────────────────────────────────────────────────────

def measurement_noise(ground: "GroundPlane", feet_px, box_h=None,
                      scale=None) -> np.ndarray:
    """
    Covariance (N, 2, 2) in m² of people's feet on the ground. The detector
    places the feet to within a few % of the box height (more vertically,
    where the box bottom is fuzzy); the ground plane's local Jacobian turns
    that into metres, so a far-away person, whose pixel spans metres of
    ground, gets a large, depth-stretched uncertainty. `scale` inflates it
    per person (e.g. feet guessed from a head).
    """
    p = np.asarray(feet_px, float).reshape(-1, 2)
    n = len(p)
    bh = np.full(n, np.nan) if box_h is None else np.asarray(box_h, float).reshape(n)
    bh = np.where(np.isfinite(bh), bh, 60.0)
    su = np.maximum(Config.FEET_JITTER_PX, Config.FEET_JITTER_X * bh)
    sv = np.maximum(Config.FEET_JITTER_PX, Config.FEET_JITTER_Y * bh)
    if scale is not None:
        s = np.asarray(scale, float).reshape(n)
        su, sv = su * s, sv * s
    w0 = ground.to_world(p)
    Jx = ground.to_world(p + [1.0, 0.0]) - w0            # metres per px along u
    Jy = ground.to_world(p + [0.0, 1.0]) - w0            # … and along v
    J  = np.stack([Jx, Jy], axis=2)                      # (N, 2, 2): columns = ∂world/∂(u, v)
    D  = np.zeros((n, 2, 2))
    D[:, 0, 0], D[:, 1, 1] = su ** 2, sv ** 2
    return J @ D @ J.transpose(0, 2, 1) + 1e-6 * np.eye(2)


class DigitalTwinEngine:
    def __init__(self, zone_mgr: ZoneManager, conf_est: ConfidenceEstimator,
                 frame_s: float = Config.FRAME_SKIP / Config.SOURCE_FPS):
        self.agents:    Dict[int, Agent] = {}
        self.zone_mgr   = zone_mgr
        self.conf_est   = conf_est
        self.frame_s    = frame_s      # seconds per processed frame
        self.frame_idx  = 0
        self.clock      = 0.0          # seconds
        self.motion     = MotionFilter()

    @property
    def ground(self) -> GroundPlane:
        return self.zone_mgr.ground

    def update(self, detections: list, density_per_zone: Dict[str, float],
               frame_gap: float = 1.0):
        """
        Fold one frame of detections into the twin. Each person's feet
        (fx, fy) are mapped to metres on the ground and fed to a Kalman
        filter (MotionFilter) whose measurement noise comes from the
        perspective: position and velocity (m/s) come out smoothed, with the
        velocity's uncertainty, so pixel jitter far from the camera does not
        turn into fake speed. frame_gap is the time since the previous
        processed frame, in processed frames.
        """
        self.frame_idx += 1
        self.clock     += max(frame_gap, 1e-6) * self.frame_s
        seen  = set()
        feet  = [[d["fx"], d["fy"]] for d in detections]
        world = self.ground.to_world(feet)
        if detections:
            R = measurement_noise(self.ground, feet, [d.get("bh", np.nan) for d in detections],
                                  [d.get("noise_scale", 1.0) for d in detections])
            pos, vel, vstd = self.motion.step(self.clock, [d["id"] for d in detections], world, R)
            walk = walking_weight(np.linalg.norm(vel, axis=1), vstd)
        for i, det in enumerate(detections):
            aid   = det["id"]
            xn, yn = det["fx"], det["fy"]
            seen.add(aid)
            zone  = self.zone_mgr.assign(xn, yn)
            zd    = density_per_zone.get(zone, 0.0)
            conf  = self.conf_est.update(aid, det["confidence"], zone, zd)

            a = self.agents.get(aid)
            if a is None:
                a = self.agents[aid] = Agent(id=aid, x=xn, y=yn)
            a.x, a.y = xn, yn
            a.wx, a.wy = float(pos[i, 0]), float(pos[i, 1])
            a.vx, a.vy = float(vel[i, 0]), float(vel[i, 1])
            a.speed    = math.hypot(a.vx, a.vy)
            a.vel_std  = float(vstd[i])
            a.walking  = float(walk[i])
            a.zone       = zone
            a.confidence = conf
            a.last_seen  = self.clock
            a.missed     = 0

        # Lost tracks stay in the twin for TRACKER_BUFFER frames (ByteTrack may
        # re-acquire them) and are purged after that.
        gone = []
        for aid in set(self.agents) - seen:
            a = self.agents[aid]
            a.missed += 1
            if a.missed > Config.TRACKER_BUFFER:
                self.conf_est.drop(aid)
                del self.agents[aid]
                gone.append(aid)
        self.motion.drop(gone)

    def active_agents(self) -> List[Agent]:
        """Agents detected in the current frame."""
        return [a for a in self.agents.values() if a.missed == 0]

    def set_ground(self, zone_mgr: ZoneManager):
        """Switch to a new ground calibration; tracks restart in the new metres."""
        self.zone_mgr = zone_mgr
        self.motion.reset()
        for a in self.agents.values():
            a.wx, a.wy = map(float, self.ground.to_world([[a.x, a.y]])[0])
            a.vx = a.vy = a.speed = a.walking = 0.0


# ─── Density & Flow Estimator ─────────────────────────────────────────────────

class DensityFlowEstimator:
    def __init__(self, zone_mgr: ZoneManager):
        self.zone_mgr = zone_mgr

    def density_map(self, agents: List[Agent], h: int, w: int,
                    scale: int = 8) -> np.ndarray:
        """
        Crowd-density heatmap in [0, 1]. Each person is a Gaussian
        HEATMAP_SIGMA_M metres wide on the ground, so people far away get
        smaller blobs and a dense crowd in the distance still reads as dense.
        Built at 1/scale resolution in a few blob-size bands, then upsampled.
        """
        hs, ws = max(1, h // scale), max(1, w // scale)
        d = np.zeros((hs, ws), dtype=np.float32)
        if agents:
            pts   = np.array([[a.x, a.y] for a in agents])
            sigma = Config.HEATMAP_SIGMA_M / self.zone_mgr.ground.m_per_px(pts) / scale
            sigma = np.clip(sigma, 0.5, max(hs, ws) / 4)
            rows  = np.clip((pts[:, 1] / scale).astype(int), 0, hs - 1)
            cols  = np.clip((pts[:, 0] / scale).astype(int), 0, ws - 1)
            bands = np.round(np.log(sigma) / np.log(1.5))
            for band in np.unique(bands):
                sel   = bands == band
                layer = np.zeros_like(d)
                np.add.at(layer, (rows[sel], cols[sel]), 1.0)
                s = float(np.median(sigma[sel]))
                d += gaussian_filter(layer, sigma=s) * s * s     # same peak for every person
        if d.max() > 0:
            d /= d.max()
        return cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)

    def zone_density(self, agents: List[Agent]) -> Dict[str, float]:
        """Persons per m² of ground in each zone (0 where it has no ground in range)."""
        counts = self.zone_mgr.count(agents)
        return {
            name: counts[name] / z.area_m2 if z.area_m2 >= 1.0 else 0.0
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
            "slope":       round(float(sl), 4) if np.isfinite(sl) else 0.0,
            "r2":          round(float(r ** 2), 3) if np.isfinite(r) else 0.0,   # constant counts
        }


# ─── Simulation Trigger ───────────────────────────────────────────────────────

class SimulationTrigger:
    """
    Layer J — runs the short-horizon simulation only when it is needed: the
    crowd is GROWING, the scene risk is MEDIUM/HIGH, or the video and IoT
    streams disagree (fusion confidence κ below SIM_FUSION_CONF).
    """
    def __init__(self, fusion_conf_thr: float = Config.SIM_FUSION_CONF):
        self.fusion_conf_thr = fusion_conf_thr

    def should_trigger(self, trend: str, risk_label: str, fusion_conf: float) -> bool:
        return bool(self.reasons(trend, risk_label, fusion_conf))

    def reasons(self, trend: str, risk_label: str, fusion_conf: float) -> List[str]:
        """Why the forecast is needed now (empty when it isn't)."""
        out = []
        if trend == "GROWING":
            out.append("crowd growing")
        if risk_label in ("HIGH", "MEDIUM"):
            out.append(f"{risk_label} risk")
        if fusion_conf < self.fusion_conf_thr:
            out.append("camera and gates disagree")
        return out


# ─── Risk Estimator ───────────────────────────────────────────────────────────

class RiskEstimator:
    """
    Hybrid confidence-weighted risk score.
    Paper formula:
      R = 0.40×density_norm + 0.30×speed_norm + 0.20×trend_score + 0.10×(1−confidence)
    Density is normalised by 0.8 persons/m² (Fruin LOS F), speed by 3 m/s.
    """
    W = {"density": 0.40, "speed": 0.30, "trend": 0.20, "confidence": 0.10}
    MEDIUM_THR = 0.33
    HIGH_THR   = 0.66

    def classify(self, density: float, speed: float,
                 trend: str, confidence: float = 1.0) -> dict:
        d  = float(np.clip(density / (Config.DENSITY_HIGH * 2), 0, 1))
        s  = float(np.clip(speed   / (Config.SPEED_HIGH    * 2), 0, 1))
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

    FORECAST = "FORECAST: {zone} predicted to reach capacity ({cap}) in {secs:.0f} s."

    def generate(self, zone_risks: dict, zone_counts: dict,
                 forecast_full: Optional[Dict[str, float]] = None) -> List[str]:
        """
        Alerts for zones at MEDIUM/HIGH risk now, and — from the 20 s
        forecast — zones not yet full that are predicted to reach capacity.
        """
        alerts = []
        for zone, r in zone_risks.items():
            if r["risk_label"] in ("HIGH", "MEDIUM"):
                alerts.append(self.MSGS[r["risk_label"]].format(
                    zone=zone, count=zone_counts.get(zone, 0)
                ))
        for zone, (secs, cap) in sorted((forecast_full or {}).items(), key=lambda kv: kv[1][0]):
            if 0 < secs <= Config.FORECAST_ALERT_MAX_S:
                alerts.append(self.FORECAST.format(zone=zone, cap=cap, secs=secs))
        return alerts

# ─── FBOX + HBOX Fusion ──────────────────────────────────────

def box_center(box):
    x1, y1, x2, y2 = box
    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


def point_inside_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box

    return (
        x1 <= px <= x2 and
        y1 <= py <= y2
    )


def head_matches_body(head_box, body_box):
    """
    Determine whether an HBOX belongs to an FBOX.

    We use head-center containment instead of IoU because
    a head box is much smaller than a full-body box.
    """
    head_center = box_center(head_box)

    return point_inside_box(
        head_center,
        body_box
    )
def fuse_fbox_hbox(fbox_detections, hbox_detections):
    """
    FBOX has priority.

    If an HBOX lies inside an FBOX, it is considered
    the same person and is discarded.

    If an HBOX does not belong to any FBOX, it is
    treated as a person missed by the FBOX detector.
    """

    fused = []

    # ---------------------------------------------------------
    # 1. FBOX detections have priority
    # ---------------------------------------------------------

    for det in fbox_detections:

        fused.append({
            **det,
            "source": "fbox"
        })

    # ---------------------------------------------------------
    # 2. Add only unmatched HBOX detections
    # ---------------------------------------------------------

    for hbox in hbox_detections:

        matched = False

        for fbox in fbox_detections:

            if head_matches_body(
                hbox["box"],
                fbox["box"]
            ):
                matched = True
                break

        if not matched:

            fused.append({
                **hbox,
                "source": "hbox"
            })

    return fused
# ─── Video Input Handler ──────────────────────────────────────────────────────

class VideoInputHandler:
    """
    Thread-safe video capture. Reads from webcam, RTSP, or mp4.
    Recorded files play back at their own frame rate and loop, so they
    behave like a live camera (pseudo-live demo mode).
    """
    def __init__(self, source):
        self.source   = source
        self.cap:     Optional[cv2.VideoCapture] = None
        self.fps      = Config.SOURCE_FPS
        self._frame:  Optional[np.ndarray] = None
        self._seq     = 0              # number of the newest frame
        self._ts      = 0.0            # capture time of the newest frame
        self._lock    = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._is_file = isinstance(source, str) and Path(source).is_file()
        self.reconnects = 0

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        if not self._is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps and 1.0 <= fps <= 240.0:
            self.fps = float(fps)
        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[Video] Source opened: {self.source} ({self.fps:.1f} FPS)")

    def _reconnect(self):
        """A live stream stopped delivering: reopen it, backing off up to 30 s between tries."""
        delay = 1.0
        while self._running:
            print(f"[Video] No frames from {self.source}; reconnecting in {delay:.0f} s …")
            time.sleep(delay)
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = cv2.VideoCapture(self.source)
            if self.cap.isOpened():
                ok, _ = self.cap.read()
                if ok:
                    self.reconnects += 1
                    print(f"[Video] Reconnected to {self.source}")
                    return
            delay = min(delay * 2, 30.0)

    def _capture_loop(self):
        period = 1.0 / self.fps
        next_t = time.perf_counter()
        fails  = 0
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                if self._is_file:
                    # Loop recorded video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    fails += 1
                    # ~2 s of failed reads: the camera or network is gone
                    if fails >= Config.STREAM_RETRY_READS:
                        self._reconnect()
                        fails = 0
                        next_t = time.perf_counter()
                    else:
                        time.sleep(0.01)
                continue
            fails = 0
            if self._is_file:
                # Pace playback to the file's frame rate
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -period:
                    next_t = time.perf_counter()
            with self._lock:
                self._frame = frame
                self._seq  += 1
                self._ts    = time.time()

    def read(self) -> Optional[Tuple[np.ndarray, int, float]]:
        """Newest frame as (frame, frame number, capture time), or None."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy(), self._seq, self._ts

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


# ─── Track Replay ─────────────────────────────────────────────────────────────

class TrackReplay:
    """
    Recorded tracks for the twin-only mode. Reads
      • MOT CSV files (frame, id, x, y, w, h, conf, class, visibility) — ground
        truth or tracker output; size and FPS come from seqinfo.ini if present;
      • experience logs (experience/logs/*.jsonl) written by the live pipeline.
        A folder replays its most recent run; a file replays the run it
        belongs to (logs less than RUN_GAP_S apart).
    """
    RUN_GAP_S = 120

    def __init__(self, path: str, size: Optional[Tuple[int, int]] = None,
                 fps: Optional[float] = None):
        p = Path(path)
        if p.is_dir() and (p / "gt" / "gt.txt").exists():     # MOT sequence folder
            p = p / "gt" / "gt.txt"
        self.fps  = Config.SOURCE_FPS
        self.size = (Config.FRAME_HEIGHT, Config.FRAME_WIDTH)   # (h, w)
        self.inferred_size = False
        if p.suffix == ".txt":
            self.kind = "mot"
            self._load_mot(p)
        else:
            self.kind = "log"
            self._files = self._log_run(p)
            self._load_log_meta()
        if size:
            self.size, self.inferred_size = size, False
        if fps:
            self.fps = fps

    # ── MOT CSV ──────────────────────────────────────────────────────────
    def _load_mot(self, path: Path):
        data = np.loadtxt(str(path), delimiter=",", ndmin=2)
        if data.shape[1] >= 8:
            # Ground truth: scored pedestrians only; tracker output has class −1
            data = data[(data[:, 7] == -1) | ((data[:, 7] == 1) & (data[:, 6] == 1))]
        self._mot = data[np.lexsort((data[:, 1], data[:, 0]))]    # by frame, then id
        seqinfo = next((d / "seqinfo.ini" for d in (path.parent, path.parent.parent)
                        if (d / "seqinfo.ini").exists()), None)
        if seqinfo is not None:
            import configparser
            ini = configparser.ConfigParser()
            ini.read(seqinfo)
            s = ini["Sequence"]
            self.size = (int(s["imHeight"]), int(s["imWidth"]))
            self.fps  = float(s.get("frameRate", self.fps))
        elif len(self._mot):
            self.size = self._round_size(self._mot[:, 3].max() + self._mot[:, 5].max(),
                                         self._mot[:, 2].max() + self._mot[:, 4].max())
            self.inferred_size = True

    def _mot_frames(self):
        if not len(self._mot):
            return
        frames, starts = np.unique(self._mot[:, 0].astype(int), return_index=True)
        chunks = dict(zip(frames, np.split(self._mot, starts[1:])))
        for fr in range(frames[0], frames[-1] + 1):
            yield fr, [
                {"id": int(r[1]),
                 "x1": float(r[2]), "y1": float(r[3]),
                 "x2": float(r[2] + r[4]), "y2": float(r[3] + r[5]),
                 "cx": float(r[2] + r[4] / 2), "cy": float(r[3] + r[5] / 2),
                 "confidence": float(np.clip(r[6], 0, 1))}
                for r in chunks.get(fr, ())
            ]

    # ── Experience logs ──────────────────────────────────────────────────
    @staticmethod
    def _log_span(path: Path) -> Tuple[float, float]:
        """Timestamps of the first and last record in a log file."""
        with open(path, "rb") as fh:
            first = json.loads(fh.readline())["timestamp"]
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 262144))
            last = json.loads(fh.read().splitlines()[-1])["timestamp"]
        return first, last

    def _log_run(self, p: Path) -> List[Path]:
        files = sorted((p if p.is_dir() else p.parent).glob("*.jsonl"))
        if not files:
            raise SystemExit(f"No experience logs (*.jsonl) in {p}")
        runs, cur, last = [], [], None
        for f in files:
            first_ts, last_ts = self._log_span(f)
            if last is not None and first_ts - last > self.RUN_GAP_S:
                runs.append(cur)
                cur = []
            cur.append(f)
            last = last_ts
        runs.append(cur)
        if p.is_dir():
            return runs[-1]
        return next(r for r in runs if p.resolve() in [f.resolve() for f in r])

    def _log_records(self):
        for f in self._files:
            with open(f, encoding="utf8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)

    def _load_log_meta(self):
        first = next(self._log_records())
        meta  = first.get("meta", {})
        self._exact = bool(meta.get("frame_size") and meta.get("fps"))
        if self._exact:
            w, h = meta["frame_size"]
            self.size, self.fps = (int(h), int(w)), float(meta["fps"])
            return
        # Older logs don't store the frame size: infer it from how far the
        # tracks reach and the aspect ratio of a saved frame.
        max_x = max_y = 0.0
        aspect = None
        for i, rec in enumerate(self._log_records()):
            for d in rec["detections"]:
                max_x, max_y = max(max_x, d["cx"]), max(max_y, d["cy"])
            fp = rec.get("frame_path")
            if aspect is None and fp and Path(fp).exists():
                img = cv2.imread(fp)
                if img is not None:
                    aspect = img.shape[0] / img.shape[1]
            if i >= 300:
                break
        if aspect:
            w = max(max_x, max_y / aspect)
            self.size = self._round_size(w * aspect, w)
        else:
            self.size = self._round_size(max_y, max_x)
        self.inferred_size = True

    def _log_frames(self):
        for i, rec in enumerate(self._log_records()):
            # New logs carry the source frame number; old ones one record per processed frame
            fr = rec["frame_idx"] if self._exact else i * Config.FRAME_SKIP
            yield fr, [
                {"id": int(d["id"]), "cx": float(d["cx"]), "cy": float(d["cy"]),
                 "confidence": float(d.get("conf", 1.0)),
                 # Feet and box height (logs since the perspective fix)
                 **({"fy": float(d["fy"]), "bh": float(d["bh"])} if "fy" in d else {})}
                for d in rec["detections"]
            ]

    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _round_size(h: float, w: float) -> Tuple[int, int]:
        """Round an inferred frame size up to a multiple of 16 px."""
        return (max(16, int(math.ceil(h / 16.0)) * 16),
                max(16, int(math.ceil(w / 16.0)) * 16))

    def frames(self):
        """Yield (source frame number, [track dicts]) for the whole recording."""
        return self._mot_frames() if self.kind == "mot" else self._log_frames()


# ─── Digital Twin Pipeline ────────────────────────────────────────────────────

class TwinPipeline:
    """
    The digital twin without detection: takes the tracked people in a frame and
    runs fusion, the twin, density/flow, trend, risk, alerts and the
    short-horizon simulation, then builds the dashboard payload. Needs no
    PyTorch — CDTPipeline puts the detectors in front of it and
    ReplayPipeline feeds it recorded tracks.
    """
    model_name  = "replay"
    device      = "cpu"
    calib_name  = "stream"         # names the saved auto-calibration file
    source_name = "stream"         # shown on the dashboard
    mode        = "live"           # "live" (detectors) or "replay" (recorded tracks)
    dense       = False            # dense-crowd mode (CDTPipeline)
    forecast_warnings = None       # set by TwinService: zones the forecast sees filling up
    det_counts  = {}               # people found by body / head / point this frame

    def __init__(self, record_experience: bool = True):
        self.fusion      = DataFusionLayer()
        self.trend       = TrendPredictor()
        self.risk_est    = RiskEstimator()
        self.alerts      = AlertEngine()
        self.sim_trigger = SimulationTrigger()
        self._auto_ground: Optional[GroundPlane] = None
        self._auto_tried  = False
        self._feet_rows   = deque(maxlen=5000)       # recent feet rows, to find the floor's far edge
        self._floor_top: Optional[float] = None
        self.configure(Config.FRAME_HEIGHT, Config.FRAME_WIDTH, Config.SOURCE_FPS)

        # Experience buffer: collects lightweight examples and sampled frames
        # for offline retraining / curation.
        self.experience = None
        if record_experience:
            try:
                self.experience = ExperienceBuffer(
                    base_path="experience",
                    max_buffer=800,
                    save_frames=True,
                    frame_save_interval=30,
                    max_bytes=int(Config.EXPERIENCE_MAX_GB * 2**30),
                )
            except Exception:
                self.experience = None
        self._flushing = threading.Event()

        self._frame_n        = 0
        self._proc_times     = deque(maxlen=30)     # wall-clock times of processed frames
        self._last_broadcast = 0.0
        self._running        = False
        self.latest_payload: Optional[dict] = None
        self.latest_frame: Optional[np.ndarray] = None   # raw frame, for the 3D twin's floor
        self.on_air = True                       # multi-camera: only the selected one broadcasts
        self.camera_id = "cam1"
        # Latest crowd state in metres, read by the 3D twin (TwinService)
        self.twin_snapshot: Optional[dict] = None
        self._scene_cache: Optional[dict] = None

    def configure(self, h: int, w: int, source_fps: float):
        """Size the ground, zones and twin for an h × w stream at source_fps."""
        self._h, self._w  = h, w
        self.source_fps   = source_fps
        self.frame_s      = Config.FRAME_SKIP / source_fps     # seconds per processed frame
        if (h, w) != getattr(self, "_floor_size", (h, w)):   # a new view: learn its floor again
            self._feet_rows.clear()
            self._floor_top = None
        self._floor_size  = (h, w)
        ground            = self._initial_ground(h, w)
        ground.floor_top  = self._floor_top
        # Until calibrated, estimate the ground from people's heights (once per stream)
        self._calibrator  = (PedestrianCalibrator(h, w)
                             if ground.source == "flat" and Config.AUTO_CALIBRATE
                             and not self._auto_tried else None)
        self.zone_mgr     = ZoneManager(h, w, ground)
        self.conf_est     = ConfidenceEstimator()
        self.dt           = DigitalTwinEngine(self.zone_mgr, self.conf_est, self.frame_s)
        self.density      = DensityFlowEstimator(self.zone_mgr)
        self.memory       = SceneMemory(Floor(ground.boundary()))    # flow + entries, in these metres
        self._flow        = None
        self._flow_at     = -1e9
        self.obstacles    = np.zeros((0, 2))                          # learned pillars / railings
        self._obst_at     = 0.0
        self.scene_version = getattr(self, "scene_version", 0) + 1     # 3D twin rebuilds its scene

    def _initial_ground(self, h: int, w: int) -> GroundPlane:
        """
        Calibration file, else an earlier estimate for this stream (this run,
        or saved by an earlier run as calibration/auto_<name>.json), else flat.
        """
        if Config.CALIBRATION_FILE:
            return GroundPlane.from_file(Config.CALIBRATION_FILE, h, w)
        if self._auto_ground is None and not self._auto_tried and Config.AUTO_CALIBRATE \
                and Config.REUSE_AUTO_CALIBRATION:
            saved = Path("calibration") / f"auto_{self.calib_name}.json"
            if saved.exists():
                try:
                    g = GroundPlane.from_file(saved, h, w)
                    g.source = "pedestrians"                # still an estimate, not a measurement
                    self._auto_ground, self._auto_tried = g, True
                    print(f"[Calibration] Reusing {saved.as_posix()} (delete it to estimate again)")
                except (ValueError, KeyError, OSError) as e:
                    print(f"[Calibration] Ignoring {saved.as_posix()}: {e}")
        g = self._auto_ground
        if g is not None and (g.h, g.w) != (h, w) and "camera_height_m" in g.params:
            # The pipeline is first sized to a default frame, then to the stream's
            # own: the camera pose is the same, rebuild its homography for this size
            p = g.params
            g = GroundPlane.from_camera(h, w, p["camera_height_m"], p["pitch_deg"], p["hfov_deg"],
                                        source=g.source)
            self._auto_ground = g
        if g is not None and (g.h, g.w) == (h, w):
            return g
        return GroundPlane.flat(h, w)

    def set_ground(self, ground: GroundPlane):
        """Switch the twin to a new ground calibration."""
        ground.floor_top = self._floor_top
        self.zone_mgr = ZoneManager(self._h, self._w, ground)
        self.dt.set_ground(self.zone_mgr)
        self.density.zone_mgr = self.zone_mgr
        self.memory   = SceneMemory(Floor(ground.boundary()))      # new metres: learn afresh
        self._flow, self._flow_at = None, -1e9
        self.obstacles, self._obst_at = np.zeros((0, 2)), self.dt.clock
        self.scene_version += 1

    def _update_floor(self):
        """
        The image above the highest feet seen so far is walls or sky, not
        floor: move the floor's far edge up to there. It only ever grows, so a
        far area that fills up later is added, never taken away.
        """
        top = max(0.0, float(np.percentile(self._feet_rows, 1)) - 0.02 * self._h)
        if self._floor_top is not None and top >= self._floor_top - 0.02 * self._h:
            return
        self._floor_top = top if top > 0 else None
        ground = self.zone_mgr.ground
        ground.floor_top = self._floor_top
        # Same homography, new zone areas: the twin's positions stay valid
        self.zone_mgr = ZoneManager(self._h, self._w, ground)
        self.dt.zone_mgr = self.zone_mgr
        self.density.zone_mgr = self.zone_mgr
        self.memory.set_floor(Floor(ground.boundary()))              # the far edge is a wall now
        self.scene_version += 1

    def _finish_calibration(self):
        """Fit the ground from the people collected so far, and save the result."""
        ground, why = self._calibrator.fit()
        self._calibrator = None
        self._auto_tried = True
        if ground is None:
            print(f"[Calibration] Couldn't estimate the ground ({why}); keeping the flat "
                  f"{Config.SCENE_WIDTH_M:g} x {Config.SCENE_HEIGHT_M:g} m scale. "
                  f"Use --calibration to set one.")
            return
        self._auto_ground = ground
        self.set_ground(ground)
        out = Path("calibration") / f"auto_{self.calib_name}.json"
        try:
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps({**ground.save_dict(), "note": why}, indent=2),
                           encoding="utf-8")
            print(f"[Calibration] From people's heights: {why}. Saved to {out.as_posix()}")
        except OSError as e:
            print(f"[Calibration] From people's heights: {why}. (Not saved: {e})")

    def _broadcast(self, payload: dict):
        """Rate-limited WebSocket broadcast."""
        now = time.time()
        if not self.on_air:                      # another camera is the one on screen
            return
        if now - self._last_broadcast >= 1.0 / Config.WS_HZ and app_loop is not None:
            asyncio.run_coroutine_threadsafe(
                manager.broadcast(json.dumps(payload)),
                app_loop,
            )
            self._last_broadcast = now

    def twin_scene(self) -> dict:
        """
        The 3D twin's environment, set up from the calibration: the walkable
        floor in view, its boundary edges (walls where the floor ends, the
        edges of the camera's view), the zones on the ground and the camera.
        """
        if self._scene_cache and self._scene_cache["version"] == self.scene_version:
            return self._scene_cache
        zm, ground = self.zone_mgr, self.zone_mgr.ground
        polys = zm.polygons()
        self._scene_cache = {
            "type":        "scene",
            "version":     self.scene_version,
            "floor":       np.round(ground.visible_ground(), 2).tolist(),
            "edges":       ground.boundary(),
            "zones":       [{"name": n, "polygon": np.round(polys[n], 2).tolist(),
                             "area_m2": round(z.area_m2, 1), "capacity": z.capacity}
                            for n, z in zm.zones.items()],
            "camera":      ground.camera(),
            "calibration": ground.describe(),
            "obstacles":   np.round(self.obstacles, 2).tolist(),     # cell centres, metres
            "obstacle_cell": round(self.memory.cell, 2),
            "frame_size":  [self._w, self._h],
            # image → ground homography and the floor's far edge, so the twin can
            # lay the camera image on its floor
            "homography":  np.round(ground.H, 9).tolist(),
            "floor_top":   ground.floor_top,
            "w_min":       ground.w_min,
            "video":       bool(Config.SEND_FRAME),
            "horizon_s":   round(Config.PRED_HORIZON * Config.SIM_STEP_S, 1),
        }
        return self._scene_cache

    def process_tracks(self, det_list: List[dict], frame: Optional[np.ndarray] = None,
                       hbox_detections: List[dict] = (), frame_gap: float = 1.0,
                       captured_at: Optional[float] = None,
                       frame_idx: Optional[int] = None,
                       t0: Optional[float] = None,
                       t_detect: Optional[float] = None,
                       to_ref: Optional[np.ndarray] = None) -> dict:
        """
        Run one frame of tracked people (dicts with id, cx, cy, confidence and
        optionally the box x1..y2, or feet fy and box height bh) through
        Layers A and D–M and return the dashboard payload. frame_gap is the
        time since the previous processed frame (in processed frames);
        captured_at is the capture time used for the end-to-end latency.
        to_ref (3×3), from camera-motion compensation, maps this frame's
        pixels to the reference frame the twin measures in; boxes are still
        drawn where they are in this frame.
        """
        t0 = time.perf_counter() if t0 is None else t0
        t_detect = t0 if t_detect is None else t_detect
        if captured_at is None:
            captured_at = time.time()
        self._frame_n += 1
        frame_idx = self._frame_n if frame_idx is None else frame_idx
        for det in det_list:
            # People stand where their feet are: the bottom-centre of the box
            det.setdefault("fx", det["cx"])
            if "y2" in det:
                det.setdefault("fy", det["y2"])
                det.setdefault("bh", det["y2"] - det["y1"])
            else:
                det.setdefault("fy", det["cy"])       # no box: the given point
        moving = to_ref is not None and not np.allclose(to_ref, np.eye(3))
        if moving and det_list:                      # measure feet where they are on the ground
            ref = apply_transform(to_ref, [[d["fx"], d["fy"]] for d in det_list])
            for d, (fx, fy) in zip(det_list, ref):
                d["fx"], d["fy"] = float(fx), float(fy)
        for det in det_list:
            det.setdefault("zone", self.zone_mgr.assign(det["fx"], det["fy"]))

        # ── Ground calibration from people's heights (first frames only) ──
        if self._calibrator is not None:
            self._calibrator.add(det_list, hbox_detections)
            if self._calibrator.ready:
                self._finish_calibration()

        # ── Where the floor ends: above the highest feet seen ─────────────
        self._feet_rows.extend(d["fy"] for d in det_list
                               if d["confidence"] >= 0.4 and d.get("src", "body") == "body")
        if len(self._feet_rows) >= Config.FLOOR_MIN_FEET and self._frame_n % 25 == 1:
            self._update_floor()

        # ── Data Fusion ───────────────────────────────────────────────────
        fusion_result = self.fusion.fuse(det_list, frame_idx)
        det_list      = fusion_result["fused_detections"]
        fused_count   = fusion_result["fused_count"]

        # ── DT Engine update ──────────────────────────────────────────────
        z_dens_pre = self.density.zone_density(self.dt.active_agents())
        self.dt.update(det_list, z_dens_pre, frame_gap=frame_gap)

        # ── Post-update state ─────────────────────────────────────────────
        agents_now = self.dt.active_agents()
        # What the scene teaches: lanes (flow field) and where people come in
        self.memory.observe(self.dt.clock, [a.id for a in agents_now],
                            [[a.wx, a.wy] for a in agents_now],
                            [[a.vx, a.vy] for a in agents_now],
                            [a.walking for a in agents_now])
        if self.dt.clock - self._flow_at >= 1.0:           # the 3D twin forecasts once a second
            self._flow, self._flow_at = self.memory.flow_field(), self.dt.clock
        if self.dt.clock - self._obst_at >= Config.OBSTACLE_EVERY_S:
            self._obst_at = self.dt.clock
            obst = self.memory.obstacles()
            if len(obst):                                  # only where people would be seen
                g   = self.zone_mgr.ground
                img = g.to_image(obst)
                ok  = (g.person_px(img) >= Config.OBSTACLE_MIN_PERSON_PX) &                       (img[:, 0] > 0.03 * self._w) & (img[:, 0] < 0.97 * self._w) &                       (img[:, 1] < 0.97 * self._h)
                obst = obst[ok]
            if obst.shape != self.obstacles.shape or not np.allclose(obst, self.obstacles):
                if len(obst) != len(self.obstacles):
                    print(f"[Twin] {len(obst)} floor cells learned as obstacles (never walked on)")
                self.obstacles = obst
                self.scene_version += 1                        # the 3D twin redraws them
        z_dens     = self.density.zone_density(agents_now)
        z_counts   = self.zone_mgr.count(agents_now)
        z_flow     = self.density.zone_flow(agents_now)

        mean_spd  = float(np.mean([a.speed      for a in agents_now])) if agents_now else 0.0
        mean_conf = float(np.mean([a.confidence for a in agents_now])) if agents_now else 1.0
        # Risk uses the median speed, so a few mis-tracked people can't max it out
        med_spd   = float(np.median([a.speed    for a in agents_now])) if agents_now else 0.0

        # ── Trend predictor ───────────────────────────────────────────────
        self.trend.push(len(agents_now), mean_spd)
        trend_res = self.trend.analyze()

        # ── Risk estimator ────────────────────────────────────────────────
        global_dens = float(np.mean(list(z_dens.values()))) if z_dens else 0.0
        risk_res = self.risk_est.classify(
            global_dens, med_spd, trend_res["crowd_trend"], mean_conf
        )

        zone_risks = {}
        for zn in self.zone_mgr.zones:
            zs = [a.speed      for a in agents_now if a.zone == zn]
            zc = [a.confidence for a in agents_now if a.zone == zn]
            zone_risks[zn] = self.risk_est.classify(
                z_dens.get(zn, 0.0),
                float(np.median(zs)) if zs else 0.0,
                trend_res["crowd_trend"],
                float(np.mean(zc)) if zc else 1.0,
            )

        # ── Alert engine ──────────────────────────────────────────────────
        # The 3D twin's latest forecast, if it is fresh and for this scene
        fw = self.forecast_warnings
        fresh = fw is not None and fw["scene_version"] == self.scene_version and \
            time.time() - fw["at"] <= 3.0 / Config.FORECAST_HZ
        alert_msgs = self.alerts.generate(zone_risks, z_counts, fw["full"] if fresh else None)

        # ── Simulation trigger: flags the 20 s forecast (TwinService runs it) ──
        trigger_reasons = self.sim_trigger.reasons(
            trend_res["crowd_trend"],
            risk_res["risk_label"],
            fusion_result["fusion_confidence"],
        )
        sim_triggered = bool(trigger_reasons)

        # ── Heatmap overlay ───────────────────────────────────────────────
        dmap       = self.density.density_map(agents_now, self._h, self._w)
        from_ref   = np.linalg.inv(to_ref) if moving else None
        if moving:                                   # the twin's frame → where it is in this image
            dmap = cv2.warpAffine(dmap, from_ref[:2], (self._w, self._h))
        heat       = (dmap * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        # Without SEND_FRAME (or without video, in replay) no video pixels are sent
        video = Config.SEND_FRAME and frame is not None
        if video:                                   # heat only where there are people
            alpha   = (np.clip(dmap, 0, 1) * 0.55)[..., None]
            overlay = (frame * (1 - alpha) + heat_color * alpha).astype(np.uint8)
        else:
            overlay = heat_color
        full = Config.OVERLAY_DETAIL == "full"

        # ── Zones: the floor area each one covers, filled by its risk ─────
        zone_col = {"HIGH": (50, 50, 230), "MEDIUM": (0, 165, 255), "LOW": (90, 210, 90)}
        fills = overlay.copy()
        polys = []
        for z in self.zone_mgr.zones.values():
            corners = z.image_polygon
            if corners is None or len(corners) < 3:
                continue
            corners = np.asarray(corners, float)
            if moving:
                corners = apply_transform(from_ref, corners)
            pts = corners.round().astype(np.int32)
            col = zone_col.get(zone_risks[z.name]["risk_label"], (128, 128, 128))
            cv2.fillPoly(fills, [pts], col)
            polys.append((z, pts, col))
        overlay = cv2.addWeighted(fills, 0.16, overlay, 0.84, 0)
        for z, pts, col in polys:
            cv2.polylines(overlay, [pts], True, col, 2, cv2.LINE_AA)

        # ── People: a thin box and a dot at the feet; ids only in full detail ──
        for det in det_list:
            if "x1" not in det:
                continue
            x1, y1, x2, y2 = map(int, (det["x1"], det["y1"], det["x2"], det["y2"]))
            if "head" in det:                       # found only by the head: the head box
                hx1, hy1, hx2, hy2 = map(int, det["head"])
                cv2.rectangle(overlay, (hx1, hy1), (hx2, hy2), (0, 170, 255), 1)
                continue
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (235, 235, 235), 1)
            if full:
                cv2.putText(overlay, f"#{det['id']} {det['confidence']:.2f}", (x1, max(12, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (235, 235, 235), 1)
        if full and not any(d.get("src") == "point" for d in det_list):
            for hbox in hbox_detections:            # every head the head detector found
                x1, y1, x2, y2 = map(int, hbox["box"])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 1)
        for a in agents_now:
            cv2.circle(overlay, (int(a.x), int(a.y)), 2 if self.dense else 3, (0, 0, 255), -1, cv2.LINE_AA)

        # Zone labels last, on a dark tag so they stay readable over the crowd
        for z, pts, col in polys:
            txt = f"{z.name.replace('Zone_', '')}  {z_counts.get(z.name, 0)}/{z.capacity}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cx, cy = pts.mean(0).astype(int)
            cx = int(np.clip(cx, tw // 2 + 4, self._w - tw // 2 - 4))
            cy = int(np.clip(cy, th + 6, self._h - 6))
            cv2.rectangle(overlay, (cx - tw // 2 - 4, cy - th - 5), (cx + tw // 2 + 4, cy + 5), (20, 20, 20), -1)
            cv2.rectangle(overlay, (cx - tw // 2 - 4, cy - th - 5), (cx + tw // 2 + 4, cy + 5), col, 1)
            cv2.putText(overlay, txt, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

        _, buf        = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 72])
        annotated_b64 = base64.b64encode(buf).decode()

        # ── Throughput and end-to-end latency ─────────────────────────────
        t1  = time.perf_counter()
        now = time.time()
        self._proc_times.append(now)
        if len(self._proc_times) > 1:
            fps = (len(self._proc_times) - 1) / max(self._proc_times[-1] - self._proc_times[0], 1e-9)
        else:
            fps = 1.0 / max(t1 - t0, 1e-9)
        fps = round(fps, 1)
        lat = round((now - captured_at) * 1000, 1)    # capture → payload

        # ── WebSocket payload ─────────────────────────────────────────────
        payload = {
            "frame_idx":        frame_idx,
            "timestamp":        round(now, 3),
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
            "trigger_reasons":  trigger_reasons,
            "fps":              fps,
            "latency_ms":       lat,
            "timing_ms": {
                "detect":   round((t_detect - t0) * 1000, 1),
                "pipeline": round((t1 - t0) * 1000, 1),
            },
            "device":           self.device,
            "model":            self.model_name,
            "source":           self.source_name,
            "mode":             self.mode,
            "dense_mode":       self.dense,
            "camera_moving":    bool(moving),           # motion compensation active
            "detections":       dict(self.det_counts),   # people found by body / head / point
            "frame_size":       [self._w, self._h],
            "source_fps":       round(self.source_fps, 2),
            "calibration": {
                **self.zone_mgr.ground.describe(),
                "ground_m2": round(sum(z.area_m2 for z in self.zone_mgr.zones.values()), 1),
            },
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
                "iot_mode":       self.fusion.iot.mode,
            },
            "zones": {
                n: {
                    "count":      z_counts.get(n, 0),
                    "density":    round(z_dens.get(n, 0.0), 4),
                    "area_m2":    round(self.zone_mgr.zones[n].area_m2, 1),
                    "risk":       zone_risks[n]["risk_label"],
                    "risk_score": zone_risks[n]["risk_score"],
                    "flow_x":     round(z_flow.get(n, (0, 0))[0], 2),
                    "flow_y":     round(z_flow.get(n, (0, 0))[1], 2),
                    "capacity":   self.zone_mgr.zones[n].capacity,
                }
                for n in self.zone_mgr.zones
            },
            "alerts": alert_msgs[:5],
            "agents": [
                {
                    "id":    a.id,
                    "x":     round(a.x, 1),      # feet, image px
                    "y":     round(a.y, 1),
                    "wx":    round(a.wx, 2),     # feet, metres on the ground
                    "wy":    round(a.wy, 2),
                    "speed": round(a.speed, 2),  # m/s
                    "zone":  a.zone,
                    "conf":  round(a.confidence, 3),
                }
                for a in agents_now
            ],
            "image_b64": annotated_b64,
            "bounding_boxes": [
                {
                    "id":    int(det["id"]),
                    "x1":    round(float(det["x1"]), 1),
                    "y1":    round(float(det["y1"]), 1),
                    "x2":    round(float(det["x2"]), 1),
                    "y2":    round(float(det["y2"]), 1),
                    "conf":  round(float(det["confidence"]), 3),
                    "zone":  det.get("zone", ""),
                    "src":   det.get("src", "body"),
                }
                for det in det_list if "x1" in det
            ],
            "head_bounding_boxes": [
                {
                    "x1": round(float(head["box"][0]), 1),
                    "y1": round(float(head["box"][1]), 1),
                    "x2": round(float(head["box"][2]), 1),
                    "y2": round(float(head["box"][3]), 1),
                    "conf": round(float(head["confidence"]), 3),
                }
                for head in hbox_detections
            ],
            "nfr": {
                "fps_pass":     fps >= 10,
                "latency_pass": lat < 2000,
            },
        }
        self.latest_payload = payload
        if frame is not None:
            self.latest_frame = frame

        # ── Crowd state in metres for the 3D twin (one atomic swap) ───────
        zone_ix = {n: i for i, n in enumerate(self.zone_mgr.zones)}
        self.twin_snapshot = {
            "t":             now,
            "frame_idx":     frame_idx,
            "scene_version": self.scene_version,
            "ids":           [a.id for a in agents_now],
            "pos":           np.array([[a.wx, a.wy] for a in agents_now], float).reshape(-1, 2),
            "vel":           np.array([[a.vx, a.vy] for a in agents_now], float).reshape(-1, 2),
            "vel_std":       np.array([a.vel_std for a in agents_now], float),
            "speed":         [a.speed for a in agents_now],
            "flow":          self._flow,
            "arrivals":      self.memory.arrivals(self.dt.clock),
            "floor":         Floor(self.zone_mgr.ground.boundary()),
            "obstacles":     self.obstacles,
            "obstacle_r":    self.memory.cell * 0.75,
            "zone":          [zone_ix.get(a.zone, -1) for a in agents_now],
            "zones":         [{"name": n, "count": z_counts.get(n, 0),
                               "density": round(z_dens.get(n, 0.0), 3),
                               "risk": zone_risks[n]["risk_label"]}
                              for n in self.zone_mgr.zones],
            "trigger":       {"active": sim_triggered, "reasons": trigger_reasons},
            "risk":          {"label": risk_res["risk_label"], "score": risk_res["risk_score"]},
        }

        # Record experience for offline curation / retraining (non-blocking)
        try:
            if self.experience is not None:
                # Pass the raw frame (buffer handles resizing/saving)
                self.experience.record(frame_idx, frame, det_list, agents_now, payload)
                # Flush in background when buffer grows (one flush at a time)
                if self.experience.buffer_len() >= 100 and not self._flushing.is_set():
                    self._flushing.set()
                    threading.Thread(target=self._flush_experience, daemon=True).start()
        except Exception as e:
            print(f"[Experience] record failed: {e}")

        return payload

    def _flush_experience(self):
        try:
            self.experience.flush()
        finally:
            self._flushing.clear()

    def stop(self):
        self._running = False
        if self.experience is not None:
            self.experience.flush()


# ─── Replay Pipeline (twin-only mode) ─────────────────────────────────────────

class ReplayPipeline(TwinPipeline):
    """
    Twin-only mode: replays recorded tracks through the digital twin in real
    time (looping), so the twin, forecast and dashboard run without PyTorch
    or a GPU. Detection and training happen elsewhere.
    """
    mode = "replay"

    def __init__(self, path: str, size: Optional[Tuple[int, int]] = None,
                 fps: Optional[float] = None):
        self.replay = TrackReplay(path, size=size, fps=fps)
        p = Path(path)
        self.source_name = p.as_posix()
        self.calib_name = "replay_" + (p.parent.parent.name if p.name == "gt.txt" else p.stem)
        super().__init__(record_experience=False)
        h, w = self.replay.size
        note = " (inferred — pass --size WxH to override)" if self.replay.inferred_size else ""
        print(f"[Replay] {self.replay.kind} tracks from {path}: {w}x{h}{note}, "
              f"{self.replay.fps:.1f} FPS")

    def start(self):
        self.configure(*self.replay.size, self.replay.fps)
        self._running = True
        threading.Thread(target=self._replay_loop, daemon=True).start()
        print("[CDT] Twin running on replayed tracks.")

    def _replay_loop(self):
        while self._running:
            first = last = None
            t_start = time.perf_counter()
            for frame_no, dets in self.replay.frames():
                if not self._running:
                    return
                if last is not None and frame_no - last < Config.FRAME_SKIP:
                    continue
                if first is None:
                    first = frame_no
                # Play back in real time
                delay = t_start + (frame_no - first) / self.replay.fps - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                gap  = 1.0 if last is None else (frame_no - last) / Config.FRAME_SKIP
                last = frame_no
                try:
                    payload = self.process_tracks(dets, frame_gap=gap, frame_idx=frame_no)
                except Exception as e:
                    print(f"[Replay] Frame {frame_no} failed: {e}")
                    continue
                self._broadcast(payload)
            if first is None:
                print("[Replay] No frames to replay.")
                return
            # Loop the recording with a fresh twin
            self.configure(*self.replay.size, self.replay.fps)


# ─── Main CDT Pipeline ────────────────────────────────────────────────────────

class CDTPipeline(TwinPipeline):
    """Live pipeline: video → YOLO26 body + head detection → ByteTrack → twin."""

    def __init__(self, source=None, record_experience: bool = True, name: Optional[str] = None,
                 share: Optional["CDTPipeline"] = None):
        # share: another camera's pipeline whose models (and GPU) this one uses too
        # Detection needs PyTorch; the twin-only mode never imports it.
        # Keep Ultralytics off the network (update checks, telemetry); it reads
        # this once on import. Set YOLO_OFFLINE=0 to allow it.
        os.environ.setdefault("YOLO_OFFLINE", "1")
        from ultralytics import YOLO
        from ultralytics.cfg import DEFAULT_CFG_DICT
        import supervision as sv
        self._sv = sv

        self.video  = VideoInputHandler(source) if source is not None else None
        self._name_source(source)
        if name:                                 # e.g. evaluate.py, which reads frames itself
            self.calib_name = name
        self._switch_lock = threading.Lock()     # held while a frame is processed
        self.device = resolve_device(Config.DEVICE)
        self.half   = Config.HALF and self.device != "cpu"
        # Newer Ultralytics sets FP16 with `quantize`, older versions with `half`
        self._precision = {}
        if self.half:
            self._precision = ({"quantize": 16} if "quantize" in DEFAULT_CFG_DICT
                               else {"half": True})

        if share is not None:                    # another camera: same models, one at a time
            self._infer_lock = share._infer_lock
            self.fbox_model, self.hbox_model = share.fbox_model, share.hbox_model
            self.points_model, self.model_name = share.points_model, share.model_name
            self._finish_init(record_experience)
            return
        self._infer_lock = threading.Lock()

        # Load fine-tuned weights if available, else fall back to base
        # ── FBOX model ─────────────────────────────────────────────
        fbox_path = Config.YOLO_MODEL

        if not Path(fbox_path).exists():
            print(
                f"[CDT] FBOX weights not found at {fbox_path}. "
                f"Using base YOLO26s."
            )
            fbox_path = Config.YOLO_MODEL_BASE

        self.fbox_model = YOLO(fbox_path)
        self.model_name = Path(fbox_path).name

        # ── HBOX model ─────────────────────────────────────────────
        hbox_path = Config.HBOX_MODEL

        if not Path(hbox_path).exists():
            raise FileNotFoundError(
                f"HBOX weights not found at {hbox_path}"
            )

        self.hbox_model = YOLO(hbox_path)

        print(f"[CDT] FBOX model: {fbox_path}")
        print(f"[CDT] HBOX model: {hbox_path}")

        # ── Dense-crowd head points (dense.py), if weights are in place ──
        self.points_model = None
        if Config.DENSE_MODEL and Path(Config.DENSE_MODEL).exists():
            from dense import PointCounter
            try:
                self.points_model = PointCounter(Config.DENSE_MODEL, self.device,
                                                 Config.DENSE_THRESHOLD, half=self.half,
                                                 enhance=Config.DENSE_ENHANCE)
                print(f"[CDT] Dense-crowd point model: {Config.DENSE_MODEL}")
            except Exception as e:
                print(f"[CDT] Dense-crowd point model not loaded ({e}); using head boxes only.")
        print(f"[CDT] Device: {self.device}{' (FP16)' if self.half else ''}")
        self._finish_init(record_experience)

    def _finish_init(self, record_experience: bool):
        self._hbox_last: List[dict] = []
        self._points_last = (np.zeros((0, 2)), np.zeros(0))
        from dense import HeadScale
        self.head_scale = HeadScale()            # head size by image row, for head points
        self.dense = False                       # dense mode: heads at high resolution + points
        self.camera_motion = CameraMotion()
        self.det_counts = {"bodies": 0, "heads": 0, "points": 0}
        super().__init__(record_experience)

    def _name_source(self, source):
        is_file = isinstance(source, str) and Path(source).is_file()
        self.calib_name  = (Path(source).stem if is_file
                            else f"camera{source}" if isinstance(source, int) else "stream")
        self.source_name = (Path(source).name if is_file
                            else f"camera {source}" if isinstance(source, int)
                            else re.sub(r"//[^/@]*@", "//", str(source)))     # no passwords on screen

    def switch_source(self, source):
        """
        Switch to another video input: open it, then start a fresh twin on it
        and estimate its ground again (the old calibration belongs to the old
        camera). Raises RuntimeError if the source can't be opened.
        """
        new = VideoInputHandler(source)
        new.start()
        with self._switch_lock:
            old, self.video = self.video, new
            if old is not None:
                old.stop()
            self._name_source(source)
            Config.CALIBRATION_FILE = None            # --calibration was for the first source
            self._auto_ground, self._auto_tried = None, False
            self._feet_rows.clear()
            self._floor_top = None
            self.fusion = DataFusionLayer()
            self.trend  = TrendPredictor()
            self._proc_times.clear()
            self._hbox_last = []
            self._points_last = (np.zeros((0, 2)), np.zeros(0))
            self.head_scale = type(self.head_scale)()
            self.dense = False
            self.camera_motion = CameraMotion()
            h, w = fit_within(*new.resolution, Config.FRAME_HEIGHT, Config.FRAME_WIDTH)
            self.configure(h, w, new.fps)
        if self.experience is not None and not self._flushing.is_set():
            self._flushing.set()
            threading.Thread(target=self._flush_experience, daemon=True).start()
        print(f"[CDT] Switched to {self.source_name} at {w}x{h}.")

    def configure(self, h: int, w: int, source_fps: float):
        super().configure(h, w, source_fps)
        self.tracker = self._sv.ByteTrack(
            track_activation_threshold=Config.YOLO_CONF,
            lost_track_buffer=Config.TRACKER_BUFFER,
            minimum_matching_threshold=0.8,
            frame_rate=25,
        )

    def start(self):
        self.video.start()
        h, w = fit_within(*self.video.resolution, Config.FRAME_HEIGHT, Config.FRAME_WIDTH)
        self.configure(h, w, self.video.fps)
        self._running = True
        threading.Thread(target=self._pipeline_loop, daemon=True).start()
        print(f"[CDT] Pipeline running at {w}x{h}.")

    def detect(self, model, frame: np.ndarray, conf: float, imgsz: int) -> List[dict]:
        """Boxes for one frame as [{"box": [x1, y1, x2, y2], "confidence": c}, ...]."""
        with self._infer_lock:                   # the models may be shared by several cameras
            res = model(frame, classes=[0], conf=conf, iou=Config.YOLO_IOU, imgsz=imgsz,
                        max_det=Config.MAX_DET, device=self.device, verbose=False,
                        **self._precision)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        return [{"box": b.tolist(), "confidence": float(c)} for b, c in zip(boxes, confs)]

    def _pipeline_loop(self):
        last_seq: Optional[int] = None
        video = self.video

        while self._running:
            if self.video is not video:          # source switched: frame numbers restart
                video, last_seq = self.video, None
            item = video.read()
            # Wait for a frame FRAME_SKIP source frames after the last one
            if item is None or (last_seq is not None and item[1] - last_seq < Config.FRAME_SKIP):
                time.sleep(0.002)
                continue
            frame, seq, captured_at = item
            gap      = 1.0 if last_seq is None else (seq - last_seq) / Config.FRAME_SKIP
            last_seq = seq

            with self._switch_lock:
                if self.video is not video:
                    continue
                try:
                    payload = self.process_frame(frame, frame_gap=gap,
                                                 captured_at=captured_at, frame_idx=seq)
                except Exception as e:
                    print(f"[CDT] Frame {seq} failed: {e}")
                    continue
            self._broadcast(payload)

    def process_frame(self, frame: np.ndarray, frame_gap: float = 1.0,
                      captured_at: Optional[float] = None,
                      frame_idx: Optional[int] = None) -> dict:
        """Detect and track the people in one frame, then run the twin (Layers A–M)."""
        t0 = time.perf_counter()

        # ── Resize to the working resolution ──────────────────────────────
        if frame.shape[:2] != (self._h, self._w):
            frame = cv2.resize(frame, (self._w, self._h))

        # ── Edge AI: full-body, head and (dense crowds) head-point detection ──
        fbox_detections = self.detect(
            self.fbox_model, frame, Config.YOLO_CONF, Config.YOLO_IMGSZ
        )

        # HBOX = head detector; in dense mode at high resolution for tiny heads
        alt = self.dense and Config.DENSE_ALTERNATE and self.points_model is not None
        heads_now = (self._frame_n % 2 == 0 or not self._hbox_last) if alt else (
            self.dense or self._frame_n % max(1, Config.HBOX_EVERY) == 0)
        if heads_now:
            self._hbox_last = self.detect(
                self.hbox_model, frame,
                Config.HBOX_CONF_DENSE if self.dense else Config.HBOX_CONF,
                Config.HBOX_IMGSZ_DENSE if self.dense else Config.HBOX_IMGSZ,
            )
        hbox_detections = self._hbox_last
        self.head_scale.add(hbox_detections)

        # Head points from the dense-crowd model (when loaded): every frame in
        # dense mode, else now and then as a probe — overhead and drone views
        # show heads the head detector can't see
        probe = (self.points_model is not None and not self.dense and Config.DENSE_AUTO
                 and self._frame_n % max(1, Config.DENSE_PROBE_EVERY) == 0)
        points_now = (self._frame_n % 2 == 1 or not len(self._points_last[0])) if alt else (
            self._frame_n % max(1, Config.DENSE_EVERY) == 0)
        if self.points_model is not None and (probe or (self.dense and points_now)):
            with self._infer_lock:
                self._points_last = self.points_model(frame)
        use_points = self.points_model is not None and (self.dense or probe)
        t_detect = time.perf_counter()

        # ── People with no body box: heads (or head points) ───────────────
        bodies = np.array([d["box"] for d in fbox_detections], np.float32).reshape(-1, 4)
        extra_heads, extra_conf, extra_src = self._unmatched_heads(
            bodies, hbox_detections, self._points_last if use_points else None,
            self.head_scale, crowd=self.dense)
        if Config.DENSE_AUTO:
            hb = np.array([h["box"] for h in hbox_detections], np.float32).reshape(-1, 4)
            n_x = int((~self._heads_of_bodies((hb[:, :2] + hb[:, 2:]) / 2, bodies)).sum())
            if use_points:                      # the point model sees heads the detector can't
                # counted as outside a dense crowd, so dense mode can't keep itself on
                n_x = max(n_x, int((~self._heads_of_bodies(self._points_last[0], bodies)).sum()))
            on  = n_x >= Config.DENSE_MIN_HEADS and n_x >= Config.DENSE_HEAD_RATIO * len(bodies)
            off = n_x < Config.DENSE_MIN_HEADS / 2 or n_x < 0.5 * Config.DENSE_HEAD_RATIO * len(bodies)
            if (on and not self.dense) or (off and self.dense):
                self.dense = on
                print(f"[CDT] Dense mode {'on' if on else 'off'} ({n_x} heads without a body, "
                      f"{len(bodies)} bodies)")

        # Heads become whole-person boxes: head on top, feet from the ground geometry.
        # These boxes are what the twin measures and what the camera-motion
        # estimate masks out, but in a dense crowd each one covers the heads of
        # the people in front, so the tracker follows the heads themselves.
        pseudo = self._person_boxes(extra_heads)
        n_b = len(bodies)
        xyxy = np.vstack([bodies, pseudo])
        track_xyxy = np.vstack([bodies, self._scale_boxes(extra_heads, Config.HEAD_TRACK_SCALE)])
        conf = np.concatenate([[d["confidence"] for d in fbox_detections], extra_conf]).astype(np.float32)
        # The tracker only starts tracks from confident detections: heads passed
        # their own threshold, so they are tracked at a neutral confidence
        track_conf = conf.copy()
        track_conf[n_b:] = np.maximum(track_conf[n_b:], Config.YOLO_CONF + 0.2)
        src = np.array(["body"] * n_b + list(extra_src))
        dets = self._sv.Detections(xyxy=track_xyxy.reshape(-1, 4), confidence=track_conf,
                                   class_id=np.where(src == "body", 0, 1).astype(int),
                                   data={"src": src, "conf": conf})
        self.det_counts = {"bodies": n_b, "heads": int((src == "head").sum()),
                           "points": int((src == "point").sum())}

        # ── ByteTrack ─────────────────────────────────────────────────────
        tracked = self.tracker.update_with_detections(dets)

        det_list = []
        if tracked.tracker_id is not None and len(tracked):
            boxes = tracked.xyxy.astype(np.float32)
            is_head = tracked.data["src"] != "body"
            heads = self._scale_boxes(boxes[is_head], 1.0 / Config.HEAD_TRACK_SCALE)
            boxes[is_head] = self._person_boxes(heads)
            head_of = dict(zip(np.flatnonzero(is_head), heads))
            for k, ((x1, y1, x2, y2), tid, c, s) in enumerate(zip(
                boxes.astype(float), tracked.tracker_id, tracked.data["conf"], tracked.data["src"]
            )):
                det = {
                    "id":         int(tid),
                    "x1":         x1,
                    "y1":         y1,
                    "x2":         x2,
                    "y2":         y2,
                    "cx":         (x1 + x2) / 2,
                    "cy":         (y1 + y2) / 2,
                    "confidence": float(c),
                    "src":        str(s),
                    # Feet guessed from a head are less certain than a body box's
                    "noise_scale": 1.0 if s == "body" else 1.5,
                }
                if k in head_of:
                    det["head"] = head_of[k].tolist()
                det_list.append(det)

        # ── Camera motion: measure people against the ground, not the image ──
        to_ref = None
        if Config.STABILIZE:
            to_ref = self.camera_motion.update(frame, xyxy)
            if self.camera_motion.reanchored:        # new reference: positions jump, restart motion
                self.dt.motion.reset()
                self.memory = SceneMemory(Floor(self.zone_mgr.ground.boundary()))
                print(f"[CDT] Camera {'moving' if self.camera_motion.moving else 'moved'}: "
                      f"motion compensation {'on' if self.camera_motion.moving else 're-anchored'}")

        return self.process_tracks(
            det_list, frame=frame, hbox_detections=hbox_detections,
            frame_gap=frame_gap, captured_at=captured_at, frame_idx=frame_idx,
            t0=t0, t_detect=t_detect, to_ref=to_ref,
        )

    @staticmethod
    def _heads_of_bodies(centres, bodies: np.ndarray, crowd: bool = False) -> np.ndarray:
        """
        Which heads (centres, (N, 2)) belong to a body box — the same person,
        not someone to add. Each body owns at most one head: the one nearest
        where its head should be (top centre), within its width and from a
        little above its top (heads poke out of the box) to 30% down. In a
        dense crowd (`crowd`) the other heads are other people, even inside a
        body box: someone standing behind. Elsewhere a head inside a body box
        is that person's (keeping them cost MOT17-02 half its MOTA: 0.55 → 0.29).
        Returns a boolean mask.
        """
        c = np.asarray(centres, float).reshape(-1, 2)
        owned = np.zeros(len(c), bool)
        b = np.asarray(bodies, float).reshape(-1, 4)
        if not len(c) or not len(b):
            return owned
        if not crowd:
            owned |= ((b[None, :, 0] <= c[:, None, 0]) & (c[:, None, 0] <= b[None, :, 2]) &
                      (b[None, :, 1] <= c[:, None, 1]) & (c[:, None, 1] <= b[None, :, 3])).any(axis=1)
        from scipy.optimize import linear_sum_assignment
        bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        hx, hy = (b[:, 0] + b[:, 2]) / 2, b[:, 1] + 0.08 * bh           # where the head should be
        dx = (c[None, :, 0] - hx[:, None]) / (0.5 * bw[:, None] + 1e-6)
        dy = (c[None, :, 1] - hy[:, None]) / (0.22 * bh[:, None] + 1e-6)
        ok = (np.abs(dx) <= 1) & (c[None, :, 1] >= b[:, 1, None] - 0.15 * bh[:, None]) & \
             (c[None, :, 1] <= b[:, 1, None] + 0.3 * bh[:, None])
        cost = np.where(ok, dx ** 2 + dy ** 2, 1e6)
        rows, cols = linear_sum_assignment(cost)
        owned[cols[cost[rows, cols] < 1e6]] = True
        return owned

    @staticmethod
    def _merge_points(pts: np.ndarray, scores: np.ndarray, size: np.ndarray) -> np.ndarray:
        """
        Indices of head points kept after merging points on the same head: the
        point model's rare doubles sit within a few pixels, while neighbours
        in a dense crowd can be only half a head apart, so the radius is small.
        """
        order = np.argsort(-np.asarray(scores))
        tree, keep, dropped = cKDTree(pts), [], np.zeros(len(pts), bool)
        for i in order:
            if dropped[i]:
                continue
            keep.append(i)
            dropped[tree.query_ball_point(pts[i], min(0.35 * size[i], 4.0))] = True
        return np.sort(np.array(keep, int))

    @staticmethod
    def _scale_boxes(boxes, k: float) -> np.ndarray:
        """Boxes (N, 4) scaled k times about their centres."""
        b = np.asarray(boxes, np.float32).reshape(-1, 4)
        c, half = (b[:, :2] + b[:, 2:]) / 2, (b[:, 2:] - b[:, :2]) / 2 * k
        return np.hstack([c - half, c + half])

    def _person_boxes(self, heads) -> np.ndarray:
        """Whole-person boxes for head boxes: from the top of the head to the feet."""
        heads = np.asarray(heads, np.float32).reshape(-1, 4)
        if not len(heads):
            return np.zeros((0, 4), np.float32)
        feet, _ = self.zone_mgr.ground.feet_from_heads(heads)
        half_w = np.maximum((heads[:, 2] - heads[:, 0]) * 1.1, 2.0)
        return np.column_stack([feet[:, 0] - half_w, heads[:, 1],
                                feet[:, 0] + half_w, feet[:, 1]]).astype(np.float32)

    def _unmatched_heads(self, bodies: np.ndarray, heads: List[dict], points=None,
                         head_scale=None, crowd: bool = False):
        """
        Heads of people the body detector missed, as head boxes (N, 4), their
        confidences and where they came from ("head" / "point"). With head
        points (dense model) the points are used — they find far more heads —
        each given a head box sized for its row by `head_scale` (dense.HeadScale,
        learned from the head detector), else by how close its neighbours are.
        Points on the same head are merged (when head sizes are known), and
        each body box claims its own head (_heads_of_bodies), so nobody is
        counted twice.
        """
        if not Config.HEADS_AS_PEOPLE or (Config.HEADS_AS_PEOPLE == "dense" and not crowd):
            return np.zeros((0, 4), np.float32), np.zeros(0), []
        if points is not None and len(points[0]):
            from dense import head_sizes
            pts, sc = np.asarray(points[0], float), np.asarray(points[1], float)
            size = head_scale.predict(pts[:, 1]) if head_scale is not None else None
            if size is not None:
                m = self._merge_points(pts, sc, size)
                pts, sc, size = pts[m], sc[m], size[m]
            s = (size if size is not None else head_sizes(pts)) / 2
            boxes = np.column_stack([pts[:, 0] - s, pts[:, 1] - s, pts[:, 0] + s, pts[:, 1] + s])
            keep = ~self._heads_of_bodies(pts, bodies, crowd)
            return boxes[keep].astype(np.float32), sc[keep], ["point"] * int(keep.sum())
        hb = np.array([h["box"] for h in heads], np.float32).reshape(-1, 4)
        keep = ~self._heads_of_bodies((hb[:, :2] + hb[:, 2:]) / 2, bodies, crowd)
        return (hb[keep], np.array([h["confidence"] for h in heads], float).reshape(-1)[keep],
                ["head"] * int(keep.sum()))

    def stop(self):
        super().stop()
        if self.video:
            self.video.stop()


# ─── 3D Digital Twin Service ──────────────────────────────────────────────────

class TwinService:
    """
    The 3D digital twin (/twin), separate from the dashboard. It reads the
    pipeline's latest crowd state in metres, streams it to the 3D viewer at
    TWIN_HZ, and runs the short-horizon forecast (Layer K) — the anticipatory
    social force model in forecast.py, PRED_HORIZON × SIM_STEP_S = 20 s ahead,
    on the walkable floor set up from the calibration — every 1/FORECAST_HZ s.
    It only works while a viewer is connected.
    """
    def __init__(self, pipeline: TwinPipeline):
        self.pipeline  = pipeline
        self.simulator = CrowdForecaster(ForecastParams(steps=Config.PRED_HORIZON,
                                                        step_s=Config.SIM_STEP_S,
                                                        substeps=Config.SIM_SUBSTEPS))
        self.latest_forecast: Optional[dict] = None
        self._scene_sent = -1
        self._running    = False
        # The twin scores its own forecasts against what then happens (per scene)
        self.skill       = ForecastSkill(Config.PRED_HORIZON, Config.SIM_STEP_S)
        self._skill_scene = None
        self._observed_t  = None

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _zone_counts_now(self, snap: dict) -> np.ndarray:
        zm = self.pipeline.zone_mgr
        ix = zm.index_many(zm.ground.to_image(snap["pos"])) if len(snap["pos"]) else np.zeros(0, int)
        return np.bincount(ix[ix >= 0], minlength=len(zm.zones)).astype(float)

    def _track_skill(self, snap: dict):
        zones = id(self.pipeline.zone_mgr)
        if zones != self._skill_scene:                           # new zones / metres
            self.skill = ForecastSkill(Config.PRED_HORIZON, Config.SIM_STEP_S)
            self._skill_scene = zones
        if snap["t"] != self._observed_t:
            self.skill.observe(snap["t"], self._zone_counts_now(snap))
            self._observed_t = snap["t"]

    def stop(self):
        self._running = False

    def _send(self, msg: dict):
        if app_loop is not None and self.pipeline.on_air:
            asyncio.run_coroutine_threadsafe(twin_manager.broadcast(json.dumps(msg)), app_loop)

    def _loop(self):
        next_state = next_forecast = 0.0
        while self._running:
            snap = self.pipeline.twin_snapshot
            viewers = bool(twin_manager.active) and self.pipeline.on_air
            # Layer J → K: a flagged scene is forecast even with no one watching
            flagged = Config.FORECAST_ALERTS and snap is not None and snap["trigger"]["active"]
            if snap is not None and (viewers or flagged):
                self._track_skill(snap)
                if viewers and snap["scene_version"] != self._scene_sent:
                    self._send(self.pipeline.twin_scene())
                    self._scene_sent = snap["scene_version"]
                if viewers and time.time() >= next_state:
                    self._send(self.state(snap))
                    next_state = time.time() + 1.0 / Config.TWIN_HZ
                if time.time() >= next_forecast:
                    try:
                        self.latest_forecast = self.forecast(snap, record=True)
                        self.pipeline.forecast_warnings = {
                            "at": time.time(), "scene_version": snap["scene_version"],
                            "full": {z["name"]: (z["full_at_s"], z["capacity"])
                                     for z in self.latest_forecast["zones"]
                                     if z["full_at_s"] is not None}}
                        if viewers:
                            self._send(self.latest_forecast)
                    except Exception as e:
                        print(f"[Twin] Forecast failed: {e}")
                    next_forecast = time.time() + 1.0 / Config.FORECAST_HZ
            time.sleep(0.02)

    @staticmethod
    def state(snap: dict) -> dict:
        """The crowd now: [id, x, y, vx, vy, speed, zone] per person (metres, m/s)."""
        agents = [
            [i, round(p[0], 2), round(p[1], 2), round(v[0], 2), round(v[1], 2), round(s, 2), z]
            for i, p, v, s, z in zip(snap["ids"], snap["pos"].tolist(), snap["vel"].tolist(),
                                     snap["speed"], snap["zone"])
        ]
        return {"type": "state", "t": round(snap["t"], 3), "frame_idx": snap["frame_idx"],
                "scene_version": snap["scene_version"], "agents": agents,
                "zones": snap["zones"], "trigger": snap["trigger"], "risk": snap["risk"]}

    def forecast(self, snap: dict, record: bool = False) -> dict:
        """
        Layer K: the crowd 20 s ahead (forecast.CrowdForecaster) from each
        person's filtered position and velocity: walkers follow their heading
        and the scene's learned lanes, standing people stay, people avoid
        those they are about to meet, leave through the edges of the view and
        slide along walls, and newcomers arrive at the learned rate. Returns
        each person's path (with the step they enter / leave), and each
        zone's predicted count and density per step — calibrated by how well
        the twin's earlier forecasts did (ForecastSkill) — and when it would
        reach its capacity. `record` keeps the forecast for that scoring.
        """
        t0 = time.perf_counter()
        zm = self.pipeline.zone_mgr
        ground = zm.ground
        steps = Config.PRED_HORIZON
        res = self.simulator.run(snap["pos"], snap["vel"], snap.get("vel_std"),
                                 floor=snap.get("floor"), flow=snap.get("flow"),
                                 arrivals=snap.get("arrivals", (0.0, [])),
                                 obstacles=snap.get("obstacles"),
                                 obstacle_r=snap.get("obstacle_r", 0.75))
        paths, n = res.paths, len(res.paths)
        active = res.active()
        zone_ix = zm.index_many(ground.to_image(paths.reshape(-1, 2))).reshape(n, steps + 1)
        zone_ix = np.where(active, zone_ix, -1)
        ids = list(snap["ids"]) + [-(k + 1) for k in range(n - res.n_tracked)]
        raw = np.stack([(zone_ix == k).sum(axis=0) for k in range(len(zm.zones))], axis=1) \
            .astype(float)                                   # (steps + 1, zones)
        now = self._zone_counts_now(snap)
        cal = np.maximum(self.skill.calibrate(now, raw), 0.0)
        if record:
            self.skill.add(snap["t"], now, raw)
        zones = []
        for k, (name, z) in enumerate(zm.zones.items()):
            counts = cal[:, k]
            dens   = counts / z.area_m2 if z.area_m2 >= 1.0 else np.zeros(steps + 1)
            full   = np.nonzero(counts >= z.capacity)[0]
            zones.append({
                "name":         name,
                "counts":       np.round(counts, 1).tolist(),
                "counts_agents": raw[:, k].astype(int).tolist(),   # before calibration
                "density":      np.round(dens, 3).tolist(),
                "capacity":     z.capacity,
                "full_at_s":    round(float(full[0]) * Config.SIM_STEP_S, 1) if len(full) else None,
            })
        return {
            "type":          "forecast",
            "t0":            round(snap["t"], 3),
            "frame_idx":     snap["frame_idx"],
            "scene_version": snap["scene_version"],
            "step_s":        Config.SIM_STEP_S,
            "steps":         steps,
            "ids":           ids,
            "paths":         np.round(paths.reshape(n, -1), 2).tolist(),
            "enter_step":    res.enter_step.tolist(),     # arrivals appear at this step
            "exit_step":     res.exit_step.tolist(),      # left the view at this step (−1: stays)
            "n_tracked":     res.n_tracked,
            "zones":         zones,
            "skill":         self.skill.report(),         # the twin's recent forecast errors
            "compute_ms":    round((time.perf_counter() - t0) * 1000, 1),
        }


    def whatif(self, snap: dict, level: float = 0.8) -> dict:
        """
        What if this crowd panicked now (forecast.CrowdForecaster.panic)?
        Not a prediction: an escape-panic scenario over WHATIF_STEPS steps —
        people's paths, how fast the area empties, and where local density
        passes 5 persons/m² (crush risk). Same shape as a forecast, so the 3D
        twin can play it on the same timeline.
        """
        t0 = time.perf_counter()
        zm = self.pipeline.zone_mgr
        steps = Config.WHATIF_STEPS
        res = self.simulator.panic(snap["pos"], snap["vel"], snap.get("vel_std"),
                                   floor=snap.get("floor"), obstacles=snap.get("obstacles"),
                                   obstacle_r=snap.get("obstacle_r", 0.75), level=level,
                                   steps=steps, step_s=Config.SIM_STEP_S)
        n = len(res.paths)
        active = res.active()
        zone_ix = zm.index_many(zm.ground.to_image(res.paths.reshape(-1, 2))).reshape(n, steps + 1) \
            if n else np.zeros((0, steps + 1), int)
        zone_ix = np.where(active, zone_ix, -1)
        zones = []
        for k, (name, z) in enumerate(zm.zones.items()):
            counts = (zone_ix == k).sum(axis=0)
            dens = counts / z.area_m2 if z.area_m2 >= 1.0 else np.zeros(steps + 1)
            local = np.where(zone_ix == k, res.density, 0.0)
            full = np.nonzero(counts >= z.capacity)[0]
            zones.append({"name": name, "counts": counts.tolist(), "density": np.round(dens, 3).tolist(),
                          "capacity": z.capacity, "max_local_density": round(float(local.max()), 2) if n else 0.0,
                          "full_at_s": round(float(full[0]) * Config.SIM_STEP_S, 1) if len(full) else None})
        danger = np.argwhere(active & (res.density >= res.danger_density))          # (person, step)
        if len(danger) > 3000:
            danger = danger[np.linspace(0, len(danger) - 1, 3000).astype(int)]
        return {
            "type": "whatif", "scenario": "panic", "level": round(float(level), 2),
            "t0": round(snap["t"], 3), "frame_idx": snap["frame_idx"],
            "scene_version": snap["scene_version"], "step_s": Config.SIM_STEP_S, "steps": steps,
            "ids": list(snap["ids"]), "paths": np.round(res.paths.reshape(n, -1), 2).tolist(),
            "enter_step": [0] * n, "exit_step": res.exit_step.tolist(), "n_tracked": n,
            "zones": zones, "summary": res.summary(),
            "danger": [[int(k), *np.round(res.paths[i, k], 2).tolist()] for i, k in danger],
            "compute_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


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
manager  = ConnectionManager()               # dashboard (/ws)
twin_manager = ConnectionManager()           # 3D twin (/ws/twin)
pipeline: Optional[TwinPipeline] = None
twin_service: Optional[TwinService] = None
# Multi-camera: one pipeline + twin service per camera; `pipeline` / `twin_service`
# are the camera on screen (every single-camera endpoint serves that one)
cameras: Dict[str, TwinPipeline] = {}
twin_services: Dict[str, TwinService] = {}
site: Optional[SiteFusion] = None
app_loop: Optional[asyncio.AbstractEventLoop] = None
STARTED_AT = time.time()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global app_loop, twin_service
    app_loop = asyncio.get_event_loop()
    for cid, p in (cameras.items() if cameras else ([("cam1", pipeline)] if pipeline else [])):
        p.start()
        twin_services[cid] = TwinService(p)
        twin_services[cid].start()
    if pipeline is not None:
        twin_service = twin_services.get(pipeline.camera_id) or next(iter(twin_services.values()), None)


@app.on_event("shutdown")
async def shutdown():
    for ts in twin_services.values():
        ts.stop()
    for p in (cameras.values() if cameras else ([pipeline] if pipeline else [])):
        p.stop()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


@app.websocket("/ws/twin")
async def ws_twin(ws: WebSocket):
    """The 3D twin's stream: the scene (on connect and when it changes), the
    crowd state and the 20 s forecast. Send "scene" to get the scene again,
    "whatif:0.8" for a panic what-if at that level."""
    await twin_manager.connect(ws)
    try:
        if pipeline is not None:
            await ws.send_text(json.dumps(pipeline.twin_scene()))
            if pipeline.twin_snapshot is not None:
                await ws.send_text(json.dumps(TwinService.state(pipeline.twin_snapshot)))
            if twin_service is not None and twin_service.latest_forecast is not None:
                await ws.send_text(json.dumps(twin_service.latest_forecast))
        while True:
            msg = await ws.receive_text()
            if msg == "scene" and pipeline is not None:
                await ws.send_text(json.dumps(pipeline.twin_scene()))
            elif msg.startswith("whatif:") and twin_service is not None \
                    and pipeline is not None and pipeline.twin_snapshot is not None:
                try:
                    level = float(msg.split(":", 1)[1])
                except ValueError:
                    level = 0.8
                wi = await asyncio.to_thread(twin_service.whatif, pipeline.twin_snapshot,
                                             min(max(level, 0.0), 1.0))
                await ws.send_text(json.dumps(wi))
    except WebSocketDisconnect:
        twin_manager.disconnect(ws)


@app.get("/api/snapshot")
async def snapshot():
    """REST fallback — latest pipeline state (without image payload)."""
    if pipeline and pipeline.latest_payload:
        p = dict(pipeline.latest_payload)
        p.pop("image_b64", None)
        return p
    return {"status": "pipeline not started"}


@app.get("/api/zones")
async def zones_info():
    if pipeline:
        return {
            n: {
                "x1": z.x1, "y1": z.y1,
                "x2": z.x2, "y2": z.y2,
                "polygon_m":  np.round(z.polygon, 2).tolist() if z.polygon is not None else [],
                "polygon_px": np.round(z.image_polygon, 1).tolist() if z.image_polygon is not None else [],
                "area_m2": round(z.area_m2, 1),
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


@app.get("/api/cameras")
async def cameras_info():
    """Every camera, with its live numbers, and which one is on screen."""
    cams = cameras or ({pipeline.camera_id: pipeline} if pipeline else {})
    out = []
    for cid, p in cams.items():
        d = p.latest_payload or {}
        out.append({"id": cid, "source": p.source_name, "on_air": p is pipeline,
                    "people": d.get("n_agents"), "risk": (d.get("risk") or {}).get("risk_label"),
                    "fps": d.get("fps"), "dense": p.dense})
    return {"cameras": out, "site": site is not None}


@app.post("/api/camera")
async def select_camera(body: dict = Body(...)):
    """Put another camera on screen: {"id": "cam2"}."""
    global pipeline, twin_service
    cid = str(body.get("id", ""))
    if cid not in cameras:
        raise HTTPException(404, f"No camera {cid!r}; cameras: {', '.join(cameras) or 'none'}")
    for c, p in cameras.items():
        p.on_air = c == cid
    pipeline, twin_service = cameras[cid], twin_services.get(cid)
    return {"on_air": cid}


@app.get("/api/site")
async def site_info():
    """
    The whole site: everyone every camera sees, on one map. With a site file
    (--site) people in overlapping views are merged and counted once; without
    one the cameras are simply added up.
    """
    cams = cameras or ({pipeline.camera_id: pipeline} if pipeline else {})
    pos = {cid: (p.twin_snapshot or {}).get("pos", np.zeros((0, 2))) for cid, p in cams.items()}
    fused = (site or SiteFusion()).fuse(pos)
    fused.pop("people") if len(fused["people"]) > 2000 else None
    return fused


@app.get("/api/frame.jpg")
async def frame_jpg():
    """The latest raw camera frame (no overlays) — the 3D twin's floor texture."""
    from fastapi.responses import Response
    f = pipeline.latest_frame if pipeline is not None else None
    if f is None or not Config.SEND_FRAME:
        raise HTTPException(404, "No video frame (twin-only mode, or SEND_FRAME is off)")
    ok, buf = await asyncio.to_thread(cv2.imencode, ".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/twin")
async def twin_info():
    """The 3D twin's scene, current crowd state and a fresh 20 s forecast."""
    if pipeline is None or pipeline.twin_snapshot is None or twin_service is None:
        return {"status": "pipeline not started"}
    snap = pipeline.twin_snapshot
    return {"scene":    pipeline.twin_scene(),
            "state":    TwinService.state(snap),
            "forecast": await asyncio.to_thread(twin_service.forecast, snap)}


def video_files() -> List[Path]:
    d = Path(Config.VIDEO_DIR)
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in Config.VIDEO_EXTS) if d.is_dir() else []


@app.get("/api/sources")
async def sources():
    """The videos the live pipeline can switch to (not in replay mode)."""
    return {
        "switchable": isinstance(pipeline, CDTPipeline),
        "mode":       pipeline.mode if pipeline else None,
        "current":    pipeline.source_name if pipeline else None,
        "sources":    [{"name": p.name, "mb": round(p.stat().st_size / 2**20, 1)}
                       for p in video_files()],
        "streams":    Config.ALLOW_STREAM_URLS,
    }


def stream_source(url):
    """
    A camera index or a network stream URL the switcher may open, or None.
    Only rtsp/rtsps/http/https URLs and camera numbers — never file paths,
    so the dashboard can't be used to open files on the server.
    """
    if isinstance(url, int) or (isinstance(url, str) and url.strip().isdigit()):
        n = int(url)
        return n if 0 <= n <= 16 else None
    if isinstance(url, str):
        u = url.strip()
        if u.lower().split("://", 1)[0] in ("rtsp", "rtsps", "http", "https") and "://" in u \
                and len(u) < 500 and not any(c in u for c in "\n\r\t "):
            return u
    return None


@app.post("/api/source")
async def set_source(body: dict = Body(...)):
    """
    Switch the live pipeline: {"name": "demo.mp4"} (a video from
    /api/sources), or {"url": "rtsp://…"} / {"url": "0"} (a network stream
    or camera; Config.ALLOW_STREAM_URLS).
    """
    if not isinstance(pipeline, CDTPipeline):
        raise HTTPException(409, "Sources can't be switched in replay mode")
    if "url" in body:
        if not Config.ALLOW_STREAM_URLS:
            raise HTTPException(403, "Opening cameras / stream URLs is switched off")
        src = stream_source(body.get("url"))
        if src is None:
            raise HTTPException(422, "Give a camera number or an rtsp:// / http(s):// URL")
    else:
        match = [p for p in video_files() if p.name == body.get("name")]
        if not match:
            raise HTTPException(404, f"No video named {body.get('name')!r} in {Config.VIDEO_DIR}/")
        src = match[0].as_posix()
    try:
        await asyncio.to_thread(pipeline.switch_source, src)
    except RuntimeError as e:
        raise HTTPException(422, str(e))
    return {"current": pipeline.source_name}


@app.get("/api/health")
async def health():
    """For monitoring long live runs: is video arriving, how fast, how long up."""
    if pipeline is None:
        return {"status": "pipeline not started"}
    p = pipeline.latest_payload or {}
    video = getattr(pipeline, "video", None)
    age = round(time.time() - p["timestamp"], 1) if "timestamp" in p else None
    return {
        "status":           "ok" if age is not None and age < 10 else "stalled",
        "mode":             pipeline.mode,
        "source":           pipeline.source_name,
        "uptime_s":         round(time.time() - STARTED_AT, 1),
        "last_update_age_s": age,
        "fps":              p.get("fps"),
        "latency_ms":       p.get("latency_ms"),
        "people":           p.get("n_agents"),
        "dense_mode":       pipeline.dense,
        "stream_reconnects": getattr(video, "reconnects", 0),
        "iot":              pipeline.fusion.iot.mode,
    }


@app.post("/api/iot")
async def iot_counts(body: dict = Body(...), x_iot_token: Optional[str] = Header(None)):
    """
    Real gate-counter counts (Stream 2): {"entry": 3, "exit": 1}. The first
    call switches the fusion from simulated to real gates. Needs header
    X-IoT-Token when Config.IOT_TOKEN is set.
    """
    if Config.IOT_TOKEN and x_iot_token != Config.IOT_TOKEN:
        raise HTTPException(401, "Missing or wrong X-IoT-Token")
    if not Config.IOT_ENABLED:
        raise HTTPException(409, "IoT gates are switched off (start the server with --iot)")
    if pipeline is None:
        raise HTTPException(503, "Pipeline not started")
    try:
        entry, exit_ = parse_counts(body)
    except (ValueError, TypeError, AttributeError) as e:
        raise HTTPException(422, str(e))
    iot = pipeline.fusion.iot
    iot.push_real(entry, exit_)
    return {"mode": iot.mode, "net_count": iot.net_count}


@app.get("/api/twin/whatif")
async def twin_whatif(panic: float = 0.8):
    """Panic what-if from the current crowd (level 0–1): paths, evacuation, crush risk."""
    if pipeline is None or pipeline.twin_snapshot is None or twin_service is None:
        return {"status": "pipeline not started"}
    return await asyncio.to_thread(twin_service.whatif, pipeline.twin_snapshot,
                                   min(max(panic, 0.0), 1.0))


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    try:
        return Path("static/index.html").read_text(encoding="utf-8")
    except FileNotFoundError:
        return "<h1>Dashboard not found. Put static/index.html in place.</h1>"


@app.get("/twin", response_class=HTMLResponse)
async def twin_page():
    try:
        return Path("static/twin.html").read_text(encoding="utf-8")
    except FileNotFoundError:
        return "<h1>3D twin not found. Put static/twin.html in place.</h1>"


try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    pass


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crowd Digital Twin — Real-Time Server")
    parser.add_argument(
        "--source", action="append",
        help="Video source: 0=webcam | rtsp://... | path/to/video.mp4 (default videos/demo.mp4). "
             "Repeat for several cameras: each gets its own twin; the dashboard switches between them"
    )
    parser.add_argument("--site", metavar="JSON",
                        help="site file placing the cameras on one map (see multicam.py): people in "
                             "overlapping views are counted once in /api/site")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--device", default=Config.DEVICE,
                        help="auto | cpu | cuda:0")
    parser.add_argument("--calibration", metavar="JSON",
                        help="Ground calibration: ≥4 image↔ground points or camera "
                             "height/pitch/FOV (see calibrate.py). Default: estimate it "
                             "from people's heights")
    parser.add_argument("--hfov", type=float, default=Config.CAMERA_HFOV_DEG,
                        help="Camera's horizontal field of view in degrees, for the estimate")
    parser.add_argument("--scene-width-m", type=float, default=Config.SCENE_WIDTH_M,
                        help="Flat fallback: ground width covered by the view, in metres")
    parser.add_argument("--scene-height-m", type=float, default=Config.SCENE_HEIGHT_M,
                        help="Flat fallback: ground depth covered by the view, in metres")
    parser.add_argument("--replay", metavar="PATH",
                        help="Twin-only mode (no PyTorch/YOLO): replay recorded tracks — "
                             "experience/logs, a MOT sequence folder or a MOT CSV file")
    parser.add_argument("--size", metavar="WxH",
                        help="Frame size of the replayed tracks (default: seqinfo.ini or the logs)")
    parser.add_argument("--fps", type=float,
                        help="Frame rate of the replayed tracks")
    parser.add_argument("--body", metavar="WEIGHTS", help=f"body detector (default {Config.YOLO_MODEL}; "
                                                          f".pt, .engine or .onnx)")
    parser.add_argument("--head", metavar="WEIGHTS", help=f"head detector (default {Config.HBOX_MODEL})")
    parser.add_argument("--dense-model", metavar="WEIGHTS",
                        help=f"dense-crowd point model (default {Config.DENSE_MODEL}, if present)")
    parser.add_argument("--iot", action="store_true",
                        help="enable IoT gate counters (simulated until real counts arrive "
                             "on POST /api/iot or MQTT); off = video-only fusion")
    parser.add_argument("--mqtt", metavar="HOST[:PORT]",
                        help="MQTT broker with gate-counter messages (see iot.py); implies --iot")
    parser.add_argument("--mqtt-topic", default="cdt/gates/#")
    parser.add_argument("--iot-token", help="require this X-IoT-Token on POST /api/iot")
    args = parser.parse_args()

    Config.DEVICE           = args.device
    Config.CALIBRATION_FILE = args.calibration
    Config.CAMERA_HFOV_DEG  = args.hfov
    Config.SCENE_WIDTH_M    = args.scene_width_m
    Config.SCENE_HEIGHT_M   = args.scene_height_m
    Config.IOT_TOKEN        = args.iot_token or Config.IOT_TOKEN
    Config.IOT_ENABLED      = bool(args.iot or args.mqtt) or Config.IOT_ENABLED
    Config.YOLO_MODEL       = args.body or Config.YOLO_MODEL
    Config.HBOX_MODEL       = args.head or Config.HBOX_MODEL
    Config.DENSE_MODEL      = args.dense_model or Config.DENSE_MODEL

    if args.replay:
        size = None
        if args.size:
            w, h = (int(v) for v in args.size.lower().split("x"))
            size = (h, w)
        src = f"replay of {args.replay}"
        pipeline = ReplayPipeline(args.replay, size=size, fps=args.fps)
    else:
        srcs = [int(s) if s.isdigit() else s for s in (args.source or ["videos/demo.mp4"])]
        first = None
        for k, s_ in enumerate(srcs):
            p = CDTPipeline(s_, share=first)
            p.camera_id, p.on_air = f"cam{k + 1}", k == 0
            cameras[p.camera_id] = p
            first = first or p
        pipeline = first
        src = ", ".join(map(str, srcs))
        if args.site:
            site = SiteFusion.from_file(args.site)

    if args.mqtt:
        # Look the fusion up per message: switching source gives the pipeline a new one
        MQTTGates(args.mqtt, args.mqtt_topic,
                  lambda e, x: pipeline.fusion.iot.push_real(e, x)).start()

    print("=" * 60)
    print("  Crowd Digital Twin — Team 28 | PES University")
    print(f"  Dashboard  → http://localhost:{args.port}/")
    print(f"  3D twin    → http://localhost:{args.port}/twin")
    print(f"  WebSocket  → ws://localhost:{args.port}/ws")
    print(f"  REST API   → http://localhost:{args.port}/api/snapshot")
    print(f"  Source     → {src}")
    print("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
