"""
Multi-camera site fusion.

Each camera's twin measures people in its own ground frame (metres; X to the
right of the camera, Y away from it). A site file places every camera on one
site map:

    {"merge_radius_m": 0.6,
     "cameras": {"north": {"x": 0,  "y": 0,  "yaw_deg": 0},
                 "south": {"x": 12, "y": 30, "yaw_deg": 180}}}

x, y: where the camera's ground origin (the point under the camera) is on the
site map; yaw_deg: the direction the camera looks, counter-clockwise from the
site's +Y axis. SiteFusion maps everyone onto the site and merges people seen
by two cameras where their views overlap, so they are counted once: two
positions from different cameras closer than merge_radius_m are one person
(matched one-to-one per pair of cameras, nearest first).

Without a site file every camera is counted separately (no overlap assumed).
"""
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


class SiteFusion:
    def __init__(self, cameras: Optional[Dict[str, dict]] = None, merge_radius_m: float = 0.6):
        self.cameras = {k: {"x": float(v.get("x", 0)), "y": float(v.get("y", 0)),
                            "yaw_deg": float(v.get("yaw_deg", 0))}
                        for k, v in (cameras or {}).items()}
        self.merge_radius_m = float(merge_radius_m)

    @classmethod
    def from_file(cls, path) -> "SiteFusion":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if "cameras" not in d:
            raise ValueError(f"{path}: expected a \"cameras\" object")
        return cls(d["cameras"], d.get("merge_radius_m", 0.6))

    def to_site(self, cam: str, pos) -> np.ndarray:
        """(N, 2) camera-ground positions → site positions (metres)."""
        p = np.asarray(pos, float).reshape(-1, 2)
        c = self.cameras.get(cam)
        if c is None:
            return p.copy()
        a = math.radians(c["yaw_deg"])
        # camera +Y (away) is the yaw direction; camera +X is to its right
        fwd = np.array([-math.sin(a), math.cos(a)])
        right = np.array([math.cos(a), math.sin(a)])
        return p[:, :1] * right + p[:, 1:] * fwd + [c["x"], c["y"]]

    def fuse(self, positions: Dict[str, np.ndarray]) -> dict:
        """
        positions: camera id → (N, 2) ground positions in that camera's frame.
        Returns the site positions of the unique people, which cameras saw
        each, and the counts.
        """
        placed = {cam: self.to_site(cam, p) for cam, p in positions.items()}
        per_cam = {cam: int(len(p)) for cam, p in placed.items()}
        # Union-find over (camera, index) so a person seen by 3 cameras merges once
        keys = [(cam, i) for cam, p in placed.items() for i in range(len(p))]
        index = {k: n for n, k in enumerate(keys)}
        parent = list(range(len(keys)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        merged = 0
        cams = [c for c in placed if c in self.cameras]      # only placed cameras can overlap
        for a_i in range(len(cams)):
            for b_i in range(a_i + 1, len(cams)):
                A, B = placed[cams[a_i]], placed[cams[b_i]]
                if not len(A) or not len(B):
                    continue
                d = np.linalg.norm(A[:, None] - B[None], axis=2)
                # pairs farther apart than the radius can never match: give them a cost
                # no real match can reach, so they don't steal partners from real ones
                big = 1e6
                r, c = linear_sum_assignment(np.where(d <= self.merge_radius_m, d, big))
                for i, j in zip(r, c):
                    if d[i, j] <= self.merge_radius_m:
                        ra, rb = find(index[(cams[a_i], i)]), find(index[(cams[b_i], j)])
                        if ra != rb:
                            parent[rb] = ra
                            merged += 1
        groups: Dict[int, List] = {}
        for k, n in index.items():
            groups.setdefault(find(n), []).append(k)
        people = []
        for members in groups.values():
            pts = np.array([placed[c][i] for c, i in members])
            people.append({"pos": pts.mean(0).round(2).tolist(),
                           "cameras": sorted({c for c, _ in members})})
        return {"unique_people": len(people), "per_camera": per_cam,
                "seen_twice": merged, "people": people,
                "placed": bool(self.cameras)}
