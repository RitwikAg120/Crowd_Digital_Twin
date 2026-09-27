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
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.stats import linregress
# Experience buffer for adaptive learning (logs + sampled frames)
from experience.experience_buffer import ExperienceBuffer


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

    # Pipeline
    FRAME_SKIP        = 2          # process every Nth source frame; 1 = every frame
    FRAME_WIDTH       = 1280       # max working resolution; larger sources are downscaled
    FRAME_HEIGHT      = 720
    SOURCE_FPS        = 25.0       # used when the source does not report its frame rate

    # Ground-plane calibration (perspective). Pixels map to metres on the ground
    # through a homography taken from, in order: CALIBRATION_FILE, an estimate
    # from people's box heights (AUTO_CALIBRATE), or the flat SCENE_* scale.
    CALIBRATION_FILE  = None       # JSON: ≥4 image↔ground points, or camera height/pitch/FOV
    AUTO_CALIBRATE    = True
    CAMERA_HFOV_DEG   = 65.0       # horizontal field of view assumed by the estimate
    PERSON_HEIGHT_M   = 1.7        # average full-body height, the estimate's ruler
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
    VEL_WINDOW_S      = 1.0        # seconds of track used for each velocity estimate
    RISK_WINDOW       = 15         # frames for trend window

    # Simulation (Layers J + K)
    PRED_HORIZON      = 25         # steps
    SIM_STEP_S        = 0.8        # seconds per step → 25 × 0.8 s = 20 s ahead
    SIM_SUBSTEPS      = 4          # integration substeps per step
    SIM_FUSION_CONF   = 0.70       # also trigger when fusion confidence κ drops below this
    TRACKER_BUFFER    = 30         # processed frames a lost track stays in the twin

    # 3D digital twin (/twin) — runs the 20 s forecast, separate from the dashboard
    TWIN_HZ           = 5          # crowd-state updates to the 3D twin per second
    FORECAST_HZ       = 1          # forecasts per second, only while the twin is open

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

    # IoT simulator
    IOT_NOISE_STD     = 2.0        # std-dev of the gate miscount (persons)
    IOT_ERR_DECAY     = 0.9        # AR(1) decay of the miscount — keeps it bounded
    IOT_OCCLUSION     = 1.15       # gates also count people the camera cannot see
    IOT_SMOOTHING     = 0.1        # occupancy follows the video count at this EMA rate
    IOT_THROUGH_RATE  = 0.02       # share of the occupancy passing through the gates per tick

    # Density
    HEATMAP_SIGMA_M   = 1.0        # Gaussian per person, in metres on the ground

    # Output
    SEND_FRAME        = True       # False → heatmap without any video pixels in the payload
    WS_HZ             = 10         # max broadcasts per second
    EXPERIENCE_MAX_GB = 2.0        # oldest experience logs/frames are deleted beyond this

    # ETH/UCY path (optional — for mobility priors)
    ETH_UCY_PATH      = "dataset/eth_ucy"


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


def load_eth_ucy(path) -> Optional[np.ndarray]:
    """
    Read one ETH/UCY trajectory file as an (N, 4) array of frame, pid, x, y
    (metres). Handles the 4-column format (tab or space separated) and the
    8-column ETH obsmat format (frame, pid, x, z, y, vx, vz, vy). Returns
    None for files that are not numeric trajectory data.
    """
    try:
        data = np.loadtxt(str(path), ndmin=2)
    except ValueError:
        return None
    if data.size == 0:
        return None
    if data.shape[1] == 4:
        return data
    if data.shape[1] == 8:
        return data[:, [0, 1, 2, 4]]
    return None


# ─── Ground Plane (perspective) ───────────────────────────────────────────────

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
                                   d.get("hfov_deg", Config.CAMERA_HFOV_DEG), source="calibration")
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

    def camera(self) -> Optional[dict]:
        """Camera pose for the 3D twin: at the ground origin, looking along +Y."""
        p = self.params
        if "camera_height_m" not in p:
            return None
        return {"height_m": round(float(p["camera_height_m"]), 2),
                "pitch_deg": round(float(p["pitch_deg"]), 2),
                "hfov_deg": round(float(p["hfov_deg"]), 2)}


class PedestrianCalibrator:
    """
    Estimates the ground plane from people's full-body boxes (single-view
    metrology). In a perspective view a person's height in pixels grows
    linearly with how far below the horizon their feet are: the horizon row
    and the slope give the camera's tilt and height, for an assumed field of
    view (CAMERA_HFOV_DEG) and person height (PERSON_HEIGHT_M). The FBOX
    model predicts whole-body boxes even for partly hidden people, so boxes
    inside a crowd still measure whole people.
    """
    def __init__(self, h: int, w: int):
        self.h, self.w = h, w
        self.frames    = 0
        self._feet: List[float]    = []
        self._heights: List[float] = []

    def add(self, dets: List[dict]):
        self.frames += 1
        for d in dets:
            fy, bh = d.get("fy"), d.get("bh")
            if not bh or d.get("confidence", 1.0) < 0.4:
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
                 len(self._feet) >= Config.CALIB_MIN_BOXES) or
                self.frames >= Config.CALIB_MAX_FRAMES)

    def fit(self) -> Tuple[Optional[GroundPlane], str]:
        """The estimated ground plane (None if it can't be estimated) and why."""
        y, hp = np.array(self._feet), np.array(self._heights)
        if len(y) < 30:
            return None, f"only {len(y)} usable full-body boxes"
        # Robust line  height = a · foot_row + b  (RANSAC, then least squares)
        rng, best = np.random.default_rng(0), None
        for _ in range(300):
            i, j = rng.choice(len(y), 2, replace=False)
            if abs(y[i] - y[j]) < 0.05 * self.h:
                continue
            a   = (hp[i] - hp[j]) / (y[i] - y[j])
            inl = np.abs(hp - (hp[i] + a * (y - y[i]))) < np.maximum(0.15 * hp, 3.0)
            if best is None or inl.sum() > best.sum():
                best = inl
        if best is None or best.sum() < max(30, 0.3 * len(y)):
            return None, "box heights don't follow a single ground plane"
        y, hp = y[best], hp[best]
        a, b = np.polyfit(y, hp, 1)
        person_h = Config.PERSON_HEIGHT_M
        if a <= 0 or a * y.max() + b < 1.15 * (a * y.min() + b):
            # Hardly any perspective (distant or overhead view): one scale from person height
            s = person_h / float(np.median(hp))
            ground = GroundPlane.flat(self.h, self.w, s * self.w, s * self.h, source="pedestrians")
            return ground, f"little perspective; {s * 100:.1f} cm per pixel from {len(y)} people"
        horizon = -b / a
        if horizon > y.min() - 5:
            return None, "the estimated horizon falls below some people's feet"
        # Exact pinhole model with the assumed field of view: fit tilt and height
        f  = (self.w / 2) / math.tan(math.radians(Config.CAMERA_HFOV_DEG) / 2)
        v0 = self.h / 2
        lo, hi = [-0.3, 0.5], [1.5, 500.0]
        x0 = np.clip([math.atan2(v0 - horizon, f), person_h / a], lo, hi)
        fit = least_squares(
            lambda p: person_height_px(y, f, v0, p[0], p[1], person_h) / hp - 1,
            x0=x0, bounds=(lo, hi), loss="soft_l1", f_scale=0.1,
        )
        pitch, cam_h = fit.x
        ground = GroundPlane.from_camera(self.h, self.w, float(cam_h), math.degrees(pitch),
                                         Config.CAMERA_HFOV_DEG, source="pedestrians")
        return ground, (f"camera ≈ {cam_h:.1f} m high, tilted {math.degrees(pitch):.0f}° down "
                        f"(from {len(y)} people, {Config.CAMERA_HFOV_DEG:.0f}° field of view assumed)")


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
    confidence: float = 1.0
    zone:       str   = ""
    track:      list  = field(default_factory=list)   # (t, wx, wy) over the last VEL_WINDOW_S
    last_seen:  float = 0.0      # twin clock (s) at the last detection
    missed:     int   = 0        # consecutive processed frames without a detection


@dataclass
class ZoneState:
    name:     str
    x1:       int
    y1:       int
    x2:       int
    y2:       int
    capacity: int   = 50
    area_m2:  float = 0.0

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
    def __init__(self, h: int, w: int, ground: GroundPlane,
                 rows: int = Config.ZONE_ROWS, cols: int = Config.ZONE_COLS):
        self.zones: Dict[str, ZoneState] = {}
        self.ground = ground
        self.h, self.w = h, w
        self.rows, self.cols = rows, cols
        self.rh, self.cw = rh, cw = h // rows, w // cols
        for idx, name in enumerate(self.names(rows, cols)):
            r, c = divmod(idx, cols)
            z = ZoneState(
                name=name,
                x1=c * cw,
                y1=r * rh,
                x2=(c + 1) * cw if c < cols - 1 else w,
                y2=(r + 1) * rh if r < rows - 1 else h,
            )
            # Ground area under the zone, with perspective
            z.area_m2  = ground.area_m2([[z.x1, z.y1], [z.x2, z.y1], [z.x2, z.y2], [z.x1, z.y2]])
            # Capacity = persons at the Fruin LOS F density (2 × DENSITY_HIGH)
            z.capacity = max(10, int(z.area_m2 * Config.DENSITY_HIGH * 2))
            self.zones[name] = z

    @staticmethod
    def names(rows: int = Config.ZONE_ROWS, cols: int = Config.ZONE_COLS) -> List[str]:
        return [f"Zone_{'ABCDEFGHIJKLMNOP'[i]}" for i in range(rows * cols)]

    def assign(self, cx: float, cy: float) -> str:
        # Feet on (or just past) the frame edge belong to the edge zone
        cx = min(max(cx, 0), self.w - 1)
        cy = min(max(cy, 0), self.h - 1)
        for name, z in self.zones.items():
            if z.x1 <= cx < z.x2 and z.y1 <= cy < z.y2:
                return name
        return "Zone_X"

    def index_many(self, pts) -> np.ndarray:
        """Zone index (in names() order) for (N, 2) image points."""
        p = np.asarray(pts, float).reshape(-1, 2)
        c = np.minimum((np.clip(p[:, 0], 0, self.w - 1) // self.cw).astype(int), self.cols - 1)
        r = np.minimum((np.clip(p[:, 1], 0, self.h - 1) // self.rh).astype(int), self.rows - 1)
        return r * self.cols + c

    def polygons(self) -> Dict[str, np.ndarray]:
        """Each zone's walkable ground in view, as a polygon in metres."""
        return {
            name: self.ground.to_world(self.ground.clip(
                [[z.x1, z.y1], [z.x2, z.y1], [z.x2, z.y2], [z.x1, z.y2]]))
            for name, z in self.zones.items()
        }

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
        return "live" if self.live else "simulated"

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
      Stream 3: ETH/UCY mobility priors (zone confidence modifier)

    Key insight: video undercounts in crowds (occlusion).
    IoT overcounts slightly (re-entry, sensor noise).
    Weighted fusion + discrepancy detection corrects both.
    """

    def __init__(self):
        self.iot             = IoTSimulator()
        self.mobility: Dict[str, float] = {}   # zone_name -> confidence_weight
        self.mobility_source = "uniform"
        self._log            = deque(maxlen=300)

    def load_mobility_prior(self, eth_ucy_root: str, zone_names) -> None:
        """
        Derive per-zone confidence priors from ETH/UCY trajectories. With no
        trajectory files (the placeholder note in dataset/eth_ucy is skipped)
        every zone gets a uniform weight of 1.0.
        """
        speeds: List[float] = []
        root  = Path(eth_ucy_root)
        files = sorted(root.glob("**/*.txt")) if root.exists() else []
        for f in files:
            traj = load_eth_ucy(f)
            if traj is None:
                continue
            for pid in np.unique(traj[:, 1]):
                p = traj[traj[:, 1] == pid]
                p = p[np.argsort(p[:, 0])]
                if len(p) > 1:
                    speeds.extend(np.linalg.norm(np.diff(p[:, 2:4], axis=0), axis=1).tolist())

        if speeds:
            global_mean = float(np.mean(speeds))
            # High global flow → slightly lower zone confidence
            # (extend with per-zone matching if you have zone-labelled data)
            weight = max(0.75, 1.0 - global_mean / 100.0)
            self.mobility_source = "eth_ucy"
            print(f"[Fusion] ETH/UCY priors loaded. Global mean speed: {global_mean:.3f}")
        else:
            weight = 1.0
            self.mobility_source = "uniform"
            print(f"[Fusion] No ETH/UCY trajectories in {eth_ucy_root}. Using uniform priors.")
        self.mobility = {name: weight for name in zone_names}

    def fuse(self, video_detections: list, frame_idx: int) -> dict:
        """
        Call every frame. Returns fusion result dict consumed by DT engine.

        Paper equation:
          C_f = W_VIDEO × C_video + W_IOT × C_iot
          κ   = 1 − (|C_video − C_iot| / max(C_video, C_iot, 1))
        """
        video_count = len(video_detections)

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


# ─── Social Force Model ───────────────────────────────────────────────────────

class SocialForceModel:
    """
    Social force pedestrian model (Helbing & Molnár, 1995), used by the
    short-horizon simulation (Layer K).

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

class DigitalTwinEngine:
    ALPHA = 0.4   # EMA velocity smoothing

    def __init__(self, zone_mgr: ZoneManager, conf_est: ConfidenceEstimator,
                 frame_s: float = Config.FRAME_SKIP / Config.SOURCE_FPS):
        self.agents:    Dict[int, Agent] = {}
        self.zone_mgr   = zone_mgr
        self.conf_est   = conf_est
        self.frame_s    = frame_s      # seconds per processed frame
        self.frame_idx  = 0
        self.clock      = 0.0          # seconds

    @property
    def ground(self) -> GroundPlane:
        return self.zone_mgr.ground

    def update(self, detections: list, density_per_zone: Dict[str, float],
               frame_gap: float = 1.0):
        """
        Fold one frame of detections into the twin. Each person's feet
        (fx, fy) are mapped to metres on the ground; velocity (m/s) is the
        slope of their last VEL_WINDOW_S of positions, EMA-smoothed, which
        keeps pixel jitter from turning into fake speed far from the camera.
        frame_gap is the time since the previous processed frame, in
        processed frames.
        """
        self.frame_idx += 1
        self.clock     += max(frame_gap, 1e-6) * self.frame_s
        seen  = set()
        world = self.ground.to_world([[d["fx"], d["fy"]] for d in detections])

        for det, (wx, wy) in zip(detections, world):
            aid   = det["id"]
            xn, yn = det["fx"], det["fy"]
            seen.add(aid)
            zone  = self.zone_mgr.assign(xn, yn)
            zd    = density_per_zone.get(zone, 0.0)
            conf  = self.conf_est.update(aid, det["confidence"], zone, zd)

            a = self.agents.get(aid)
            if a is None:
                a = self.agents[aid] = Agent(id=aid, x=xn, y=yn)
            a.x, a.y, a.wx, a.wy = xn, yn, float(wx), float(wy)
            a.zone       = zone
            a.confidence = conf
            a.last_seen  = self.clock
            a.missed     = 0
            a.track.append((self.clock, a.wx, a.wy))
            while a.track[0][0] < self.clock - Config.VEL_WINDOW_S:
                a.track.pop(0)
            n = len(a.track)
            if n > 1:
                # Least-squares slope of position over time (plain Python: tracks are short)
                tm = sum(s[0] for s in a.track) / n
                tt = sum((s[0] - tm) ** 2 for s in a.track)
                if tt > 0:
                    vx = sum((s[0] - tm) * s[1] for s in a.track) / tt
                    vy = sum((s[0] - tm) * s[2] for s in a.track) / tt
                    a.vx = self.ALPHA * vx + (1 - self.ALPHA) * a.vx
                    a.vy = self.ALPHA * vy + (1 - self.ALPHA) * a.vy
                    a.speed = math.hypot(a.vx, a.vy)

        # Lost tracks stay in the twin for TRACKER_BUFFER frames (ByteTrack may
        # re-acquire them) and are purged after that.
        for aid in set(self.agents) - seen:
            a = self.agents[aid]
            a.missed += 1
            if a.missed > Config.TRACKER_BUFFER:
                self.conf_est.drop(aid)
                del self.agents[aid]

    def active_agents(self) -> List[Agent]:
        """Agents detected in the current frame."""
        return [a for a in self.agents.values() if a.missed == 0]

    def set_ground(self, zone_mgr: ZoneManager):
        """Switch to a new ground calibration; tracks restart in the new metres."""
        self.zone_mgr = zone_mgr
        for a in self.agents.values():
            a.wx, a.wy = map(float, self.ground.to_world([[a.x, a.y]])[0])
            a.vx = a.vy = a.speed = 0.0
            a.track = [(self.clock, a.wx, a.wy)]


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
            "slope":       round(float(sl), 4),
            "r2":          round(float(r ** 2), 3),
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

    def generate(self, zone_risks: dict, zone_counts: dict) -> List[str]:
        alerts = []
        for zone, r in zone_risks.items():
            if r["risk_label"] in ("HIGH", "MEDIUM"):
                alerts.append(self.MSGS[r["risk_label"]].format(
                    zone=zone, count=zone_counts.get(zone, 0)
                ))
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

    def _capture_loop(self):
        period = 1.0 / self.fps
        next_t = time.perf_counter()
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                if self._is_file:
                    # Loop recorded video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    time.sleep(0.01)
                continue
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
    model_name = "replay"
    device     = "cpu"
    calib_name = "stream"          # names the saved auto-calibration file

    def __init__(self, record_experience: bool = True):
        self.fusion      = DataFusionLayer()
        self.trend       = TrendPredictor()
        self.risk_est    = RiskEstimator()
        self.alerts      = AlertEngine()
        self.sim_trigger = SimulationTrigger()
        # Load ETH/UCY mobility priors (zone names don't depend on resolution)
        self.fusion.load_mobility_prior(Config.ETH_UCY_PATH, ZoneManager.names())
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
        # Set zone calibration from mobility priors
        for zone, weight in self.fusion.mobility.items():
            self.conf_est.set_zone_calibration(zone, weight)
        self.dt           = DigitalTwinEngine(self.zone_mgr, self.conf_est, self.frame_s)
        self.density      = DensityFlowEstimator(self.zone_mgr)
        self.scene_version = getattr(self, "scene_version", 0) + 1     # 3D twin rebuilds its scene

    def _initial_ground(self, h: int, w: int) -> GroundPlane:
        """Calibration file, else an earlier estimate for this stream, else flat."""
        if Config.CALIBRATION_FILE:
            return GroundPlane.from_file(Config.CALIBRATION_FILE, h, w)
        if self._auto_ground is not None and (self._auto_ground.h, self._auto_ground.w) == (h, w):
            return self._auto_ground
        return GroundPlane.flat(h, w)

    def set_ground(self, ground: GroundPlane):
        """Switch the twin to a new ground calibration."""
        ground.floor_top = self._floor_top
        self.zone_mgr = ZoneManager(self._h, self._w, ground)
        self.dt.set_ground(self.zone_mgr)
        self.density.zone_mgr = self.zone_mgr
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
            "frame_size":  [self._w, self._h],
            "horizon_s":   round(Config.PRED_HORIZON * Config.SIM_STEP_S, 1),
        }
        return self._scene_cache

    def process_tracks(self, det_list: List[dict], frame: Optional[np.ndarray] = None,
                       hbox_detections: List[dict] = (), frame_gap: float = 1.0,
                       captured_at: Optional[float] = None,
                       frame_idx: Optional[int] = None,
                       t0: Optional[float] = None,
                       t_detect: Optional[float] = None) -> dict:
        """
        Run one frame of tracked people (dicts with id, cx, cy, confidence and
        optionally the box x1..y2, or feet fy and box height bh) through
        Layers A and D–M and return the dashboard payload. frame_gap is the
        time since the previous processed frame (in processed frames);
        captured_at is the capture time used for the end-to-end latency.
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
            det.setdefault("zone", self.zone_mgr.assign(det["fx"], det["fy"]))

        # ── Ground calibration from people's heights (first frames only) ──
        if self._calibrator is not None:
            self._calibrator.add(det_list)
            if self._calibrator.ready:
                self._finish_calibration()

        # ── Where the floor ends: above the highest feet seen ─────────────
        self._feet_rows.extend(d["fy"] for d in det_list if d["confidence"] >= 0.4)
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
        alert_msgs = self.alerts.generate(zone_risks, z_counts)

        # ── Simulation trigger (the 20 s forecast itself runs in the 3D twin) ──
        trigger_reasons = self.sim_trigger.reasons(
            trend_res["crowd_trend"],
            risk_res["risk_label"],
            fusion_result["fusion_confidence"],
        )
        sim_triggered = bool(trigger_reasons)

        # ── Heatmap overlay ───────────────────────────────────────────────
        dmap       = self.density.density_map(agents_now, self._h, self._w)
        heat       = (dmap * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        # Without SEND_FRAME (or without video, in replay) no video pixels are sent
        overlay    = (cv2.addWeighted(frame, 0.55, heat_color, 0.45, 0)
                      if Config.SEND_FRAME and frame is not None else heat_color)

        for z in self.zone_mgr.zones.values():
            rl  = zone_risks[z.name]["risk_label"]
            col = {"HIGH": (50, 50, 220),
                    "MEDIUM": (0, 165, 255),
                    "LOW":    (80, 200, 80)}.get(rl, (128, 128, 128))
            cv2.rectangle(overlay, (z.x1, z.y1), (z.x2, z.y2), col, 2)
            cv2.putText(overlay, f"{z.name}: {rl}",
                        (z.x1 + 6, z.y1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        # ── Draw tracked full-body boxes ──────────────────────────────────

        for det in det_list:

            if "x1" not in det:
                continue

            x1, y1, x2, y2 = map(
                int,
                (det["x1"], det["y1"], det["x2"], det["y2"])
            )

            label = f"#{det['id']} {det['confidence']:.2f}"

            # Full-body box = yellow
            col = (0, 255, 255)

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                col,
                2
            )

            cv2.putText(
                overlay,
                label,
                (x1, max(12, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                col,
                1
            )
        # ── Draw HBOX detections ──────────────────────────────────────────

        for hbox in hbox_detections:

            x1, y1, x2, y2 = map(
                int,
                hbox["box"]
            )

            label = f"HEAD {hbox['confidence']:.2f}"

            # HBOX = purple
            col = (255, 0, 255)

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                col,
                2
            )

            cv2.putText(
                overlay,
                label,
                (x1, max(12, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                col,
                1
            )

        for a in agents_now:
            cv2.circle(overlay, (int(a.x), int(a.y)), 4, (0, 0, 255), -1)

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
                "mobility_prior": self.fusion.mobility_source,
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

        # ── Crowd state in metres for the 3D twin (one atomic swap) ───────
        zone_ix = {n: i for i, n in enumerate(self.zone_mgr.zones)}
        self.twin_snapshot = {
            "t":             now,
            "frame_idx":     frame_idx,
            "scene_version": self.scene_version,
            "ids":           [a.id for a in agents_now],
            "pos":           np.array([[a.wx, a.wy] for a in agents_now], float).reshape(-1, 2),
            "vel":           np.array([[a.vx, a.vy] for a in agents_now], float).reshape(-1, 2),
            "speed":         [a.speed for a in agents_now],
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
    def __init__(self, path: str, size: Optional[Tuple[int, int]] = None,
                 fps: Optional[float] = None):
        self.replay = TrackReplay(path, size=size, fps=fps)
        p = Path(path)
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

    def __init__(self, source=None, record_experience: bool = True):
        # Detection needs PyTorch; the twin-only mode never imports it
        from ultralytics import YOLO
        from ultralytics.cfg import DEFAULT_CFG_DICT
        import supervision as sv
        self._sv = sv

        self.video  = VideoInputHandler(source) if source is not None else None
        self.calib_name = (Path(source).stem if isinstance(source, str) and Path(source).is_file()
                           else f"camera{source}" if isinstance(source, int) else "stream")
        self.device = resolve_device(Config.DEVICE)
        self.half   = Config.HALF and self.device != "cpu"
        # Newer Ultralytics sets FP16 with `quantize`, older versions with `half`
        self._precision = {}
        if self.half:
            self._precision = ({"quantize": 16} if "quantize" in DEFAULT_CFG_DICT
                               else {"half": True})

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
        print(f"[CDT] Device: {self.device}{' (FP16)' if self.half else ''}")

        self._hbox_last: List[dict] = []
        super().__init__(record_experience)

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
        res = model(
            frame,
            classes=[0],
            conf=conf,
            iou=Config.YOLO_IOU,
            imgsz=imgsz,
            max_det=Config.MAX_DET,
            device=self.device,
            verbose=False,
            **self._precision,
        )[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        return [{"box": b.tolist(), "confidence": float(c)} for b, c in zip(boxes, confs)]

    def _pipeline_loop(self):
        last_seq: Optional[int] = None

        while self._running:
            item = self.video.read()
            # Wait for a frame FRAME_SKIP source frames after the last one
            if item is None or (last_seq is not None and item[1] - last_seq < Config.FRAME_SKIP):
                time.sleep(0.002)
                continue
            frame, seq, captured_at = item
            gap      = 1.0 if last_seq is None else (seq - last_seq) / Config.FRAME_SKIP
            last_seq = seq

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

        # ── Edge AI: separate full-body and head detection ───────────────
        # FBOX is the only detector used for tracking and agent state.
        fbox_detections = self.detect(
            self.fbox_model, frame, Config.YOLO_CONF, Config.YOLO_IMGSZ
        )

        # HBOX = head detector
        if self._frame_n % max(1, Config.HBOX_EVERY) == 0:
            self._hbox_last = self.detect(
                self.hbox_model, frame, Config.HBOX_CONF, Config.HBOX_IMGSZ
            )
        hbox_detections = self._hbox_last
        t_detect = time.perf_counter()

        xyxy = np.array(
            [d["box"] for d in fbox_detections],
            dtype=np.float32
        ).reshape(-1, 4)

        confidence = np.array(
            [d["confidence"] for d in fbox_detections],
            dtype=np.float32
        )

        class_id = np.zeros(
            len(fbox_detections),
            dtype=int
        )

        dets = self._sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )
        # ── ByteTrack ─────────────────────────────────────────────────────
        tracked = self.tracker.update_with_detections(dets)

        det_list = []
        if tracked.tracker_id is not None:
            for xyxy, tid, conf in zip(
                tracked.xyxy, tracked.tracker_id, tracked.confidence
            ):
                x1, y1, x2, y2 = map(float, xyxy)
                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)
                det_list.append({
                    "id":         int(tid),
                    "x1":         x1,
                    "y1":         y1,
                    "x2":         x2,
                    "y2":         y2,
                    "cx":         cx,
                    "cy":         cy,
                    "confidence": float(conf),
                })

        return self.process_tracks(
            det_list, frame=frame, hbox_detections=hbox_detections,
            frame_gap=frame_gap, captured_at=captured_at, frame_idx=frame_idx,
            t0=t0, t_detect=t_detect,
        )

    def stop(self):
        super().stop()
        if self.video:
            self.video.stop()


# ─── 3D Digital Twin Service ──────────────────────────────────────────────────

class TwinService:
    """
    The 3D digital twin (/twin), separate from the dashboard. It reads the
    pipeline's latest crowd state in metres, streams it to the 3D viewer at
    TWIN_HZ, and runs the short-horizon forecast (Layer K) — the social force
    model PRED_HORIZON × SIM_STEP_S = 20 s ahead, on the walkable floor set up
    from the calibration — every 1/FORECAST_HZ s. It only works while a viewer
    is connected.
    """
    def __init__(self, pipeline: TwinPipeline):
        self.pipeline  = pipeline
        self.simulator = SocialForceModel()
        self.latest_forecast: Optional[dict] = None
        self._scene_sent = -1
        self._running    = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _send(self, msg: dict):
        if app_loop is not None:
            asyncio.run_coroutine_threadsafe(twin_manager.broadcast(json.dumps(msg)), app_loop)

    def _loop(self):
        next_state = next_forecast = 0.0
        while self._running:
            snap = self.pipeline.twin_snapshot
            if twin_manager.active and snap is not None:
                if snap["scene_version"] != self._scene_sent:
                    self._send(self.pipeline.twin_scene())
                    self._scene_sent = snap["scene_version"]
                if time.time() >= next_state:
                    self._send(self.state(snap))
                    next_state = time.time() + 1.0 / Config.TWIN_HZ
                if time.time() >= next_forecast:
                    try:
                        self.latest_forecast = self.forecast(snap)
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

    def forecast(self, snap: dict) -> dict:
        """
        Layer K: every person 20 s ahead with the social force model, from
        their tracked position and velocity, staying on the walkable floor.
        Returns each person's path, and each zone's predicted count and
        density per step and when it would reach its capacity.
        """
        t0 = time.perf_counter()
        zm = self.pipeline.zone_mgr
        ground = zm.ground
        steps, n = Config.PRED_HORIZON, len(snap["ids"])
        paths = np.zeros((n, steps + 1, 2))
        if n:
            floor = ground.visible_ground()
            traj  = self.simulator.simulate(snap["pos"], snap["vel"], steps, Config.SIM_STEP_S,
                                            bounds=floor if len(floor) >= 3 else None)
            paths = np.stack([snap["pos"]] + traj, axis=1)
        zone_ix = zm.index_many(ground.to_image(paths.reshape(-1, 2))).reshape(n, steps + 1)
        zones = []
        for k, (name, z) in enumerate(zm.zones.items()):
            counts = (zone_ix == k).sum(axis=0)
            dens   = counts / z.area_m2 if z.area_m2 >= 1.0 else np.zeros(steps + 1)
            full   = np.nonzero(counts >= z.capacity)[0]
            zones.append({
                "name":      name,
                "counts":    counts.tolist(),
                "density":   np.round(dens, 3).tolist(),
                "capacity":  z.capacity,
                "full_at_s": round(float(full[0]) * Config.SIM_STEP_S, 1) if len(full) else None,
            })
        return {
            "type":          "forecast",
            "t0":            round(snap["t"], 3),
            "frame_idx":     snap["frame_idx"],
            "scene_version": snap["scene_version"],
            "step_s":        Config.SIM_STEP_S,
            "steps":         steps,
            "ids":           snap["ids"],
            "paths":         np.round(paths.reshape(n, -1), 2).tolist(),
            "zones":         zones,
            "compute_ms":    round((time.perf_counter() - t0) * 1000, 1),
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
app_loop: Optional[asyncio.AbstractEventLoop] = None

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
    if pipeline:
        pipeline.start()
        twin_service = TwinService(pipeline)
        twin_service.start()


@app.on_event("shutdown")
async def shutdown():
    if twin_service:
        twin_service.stop()
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


@app.websocket("/ws/twin")
async def ws_twin(ws: WebSocket):
    """The 3D twin's stream: the scene (on connect and when it changes), the
    crowd state and the 20 s forecast. Send "scene" to get the scene again."""
    await twin_manager.connect(ws)
    try:
        if pipeline is not None:
            await ws.send_text(json.dumps(pipeline.twin_scene()))
            if pipeline.twin_snapshot is not None:
                await ws.send_text(json.dumps(TwinService.state(pipeline.twin_snapshot)))
            if twin_service is not None and twin_service.latest_forecast is not None:
                await ws.send_text(json.dumps(twin_service.latest_forecast))
        while True:
            if await ws.receive_text() == "scene" and pipeline is not None:
                await ws.send_text(json.dumps(pipeline.twin_scene()))
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


@app.get("/api/twin")
async def twin_info():
    """The 3D twin's scene, current crowd state and a fresh 20 s forecast."""
    if pipeline is None or pipeline.twin_snapshot is None or twin_service is None:
        return {"status": "pipeline not started"}
    snap = pipeline.twin_snapshot
    return {"scene":    pipeline.twin_scene(),
            "state":    TwinService.state(snap),
            "forecast": await asyncio.to_thread(twin_service.forecast, snap)}


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
        "--source", default="videos/demo.mp4",
        help="Video source: 0=webcam | rtsp://... | path/to/video.mp4"
    )
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
    args = parser.parse_args()

    Config.DEVICE           = args.device
    Config.CALIBRATION_FILE = args.calibration
    Config.CAMERA_HFOV_DEG  = args.hfov
    Config.SCENE_WIDTH_M    = args.scene_width_m
    Config.SCENE_HEIGHT_M   = args.scene_height_m

    if args.replay:
        size = None
        if args.size:
            w, h = (int(v) for v in args.size.lower().split("x"))
            size = (h, w)
        src = f"replay of {args.replay}"
        pipeline = ReplayPipeline(args.replay, size=size, fps=args.fps)
    else:
        src = int(args.source) if args.source.isdigit() else args.source
        pipeline = CDTPipeline(src)

    print("=" * 60)
    print("  Crowd Digital Twin — Team 28 | PES University")
    print(f"  Dashboard  → http://localhost:{args.port}/")
    print(f"  3D twin    → http://localhost:{args.port}/twin")
    print(f"  WebSocket  → ws://localhost:{args.port}/ws")
    print(f"  REST API   → http://localhost:{args.port}/api/snapshot")
    print(f"  Source     → {src}")
    print("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
