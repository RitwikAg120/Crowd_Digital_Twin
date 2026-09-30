"""
Camera-motion compensation for handheld, panning or broadcast footage.

With a moving camera, a person standing still slides across the image and
the twin reads the camera's motion as crowd speed. CameraMotion estimates how
the image moved between processed frames — a similarity transform fitted with
RANSAC to background features tracked by pyramidal Lucas-Kanade, with
people's boxes masked out — and keeps the transform from the current frame
to a fixed reference frame. People's feet are measured in the reference
frame, so their speeds are relative to the ground.

It turns itself on only when the camera moves (the median frame-to-frame
shift over the last frames is above MOVING_PX); a fixed CCTV camera is left
alone, so its tracks don't pick up estimation noise. After a large pan or
zoom the reference no longer overlaps the view: it re-anchors to the current
frame and reports that, so the twin can restart its motion state.
"""

from collections import deque
from typing import Optional

import cv2
import numpy as np


class CameraMotion:
    WORK_WIDTH  = 640          # estimate on a downscaled grey frame
    MAX_CORNERS = 400
    MIN_INLIERS = 20
    MOVING_PX   = 0.6          # median shift (full-res px per processed frame) that means "moving" …
    STILL_PX    = 0.35         # … and "still again" (hysteresis: no flicker on a fixed camera)
    HISTORY     = 10
    REANCHOR    = 0.35         # re-anchor after drifting this share of the frame, or 30 % zoom

    def __init__(self):
        self.prev: Optional[np.ndarray] = None
        self.scale = 1.0
        self.to_ref = np.eye(3)                    # current frame → reference frame (homogeneous)
        self.shifts = deque(maxlen=self.HISTORY)
        self.moving = False
        self.reanchored = False
        self.last_inliers = 0

    def _grey(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self.scale = min(1.0, self.WORK_WIDTH / w)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return cv2.resize(g, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA) \
            if self.scale < 1 else g

    def estimate(self, prev: np.ndarray, cur: np.ndarray, boxes=None) -> Optional[np.ndarray]:
        """3×3 transform taking points in `prev` to `cur` (grey, work scale), or None."""
        mask = np.full(prev.shape, 255, np.uint8)
        for x1, y1, x2, y2 in (boxes if boxes is not None else []):
            s = self.scale
            cv2.rectangle(mask, (int(x1 * s) - 3, int(y1 * s) - 3), (int(x2 * s) + 3, int(y2 * s) + 3), 0, -1)
        if (mask > 0).mean() < 0.05:               # a frame full of people: use everything
            mask[:] = 255
        p0 = cv2.goodFeaturesToTrack(prev, self.MAX_CORNERS, 0.01, 8, mask=mask)
        if p0 is None or len(p0) < self.MIN_INLIERS:
            return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1
        if ok.sum() < self.MIN_INLIERS:
            return None
        M, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC,
                                             ransacReprojThreshold=2.0, maxIters=2000)
        self.last_inliers = int(inl.sum()) if inl is not None else 0
        if M is None or self.last_inliers < self.MIN_INLIERS:
            return None
        A = np.vstack([M, [0, 0, 1]])
        S = np.diag([self.scale, self.scale, 1.0])
        return np.linalg.inv(S) @ A @ S            # back to full-resolution pixels

    def update(self, frame: np.ndarray, boxes=None) -> np.ndarray:
        """
        Fold in a processed frame (and the people's boxes in it, to mask
        out). Returns the 3×3 current → reference transform to apply to
        image points — identity while the camera is still. `reanchored`
        says whether the reference was just reset.
        """
        cur = self._grey(frame)
        self.reanchored = False
        if self.prev is not None and self.prev.shape == cur.shape:
            A = self.estimate(self.prev, cur, boxes)
            if A is not None:
                self.shifts.append(float(np.hypot(A[0, 2], A[1, 2])))
                self.to_ref = self.to_ref @ np.linalg.inv(A)          # cur → prev → … → reference
        self.prev = cur
        was = self.moving
        med = float(np.median(self.shifts)) if self.shifts else 0.0
        if not was:
            self.moving = len(self.shifts) >= self.HISTORY and med > self.MOVING_PX
        else:
            self.moving = med > self.STILL_PX
        if self.moving and not was:
            self.to_ref = np.eye(3)                                   # start measuring from here
            self.reanchored = True
        h, w = frame.shape[:2]
        drift = np.hypot(self.to_ref[0, 2], self.to_ref[1, 2]) / max(w, h)
        zoom = abs(np.sqrt(abs(np.linalg.det(self.to_ref[:2, :2]))) - 1)
        if drift > self.REANCHOR or zoom > 0.3:
            self.to_ref = np.eye(3)
            self.reanchored = True
        return self.to_ref if self.moving else np.eye(3)


def apply(T: np.ndarray, pts) -> np.ndarray:
    """Transform (N, 2) image points by a 3×3 transform."""
    p = np.asarray(pts, float).reshape(-1, 2)
    q = p @ T[:2, :2].T + T[:2, 2]
    return q
