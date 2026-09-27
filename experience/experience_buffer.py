import json
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import cv2
import numpy as np


class ExperienceBuffer:
    """Simple experience buffer that saves lightweight metadata to JSONL
    and optionally saves sampled frames to disk for later curation / labeling.

    Usage:
      eb = ExperienceBuffer(base_path="experience", max_buffer=500)
      eb.record(frame_idx, frame, detections, agents, payload)
      eb.flush()  # writes logs to disk
    """

    def __init__(
        self,
        base_path: str = "experience",
        max_buffer: int = 500,
        save_frames: bool = True,
        frame_save_interval: int = 30,
        img_width: int = 640,
        max_bytes: int = 2 * 2**30,
    ) -> None:
        self.base = Path(base_path)
        self.logs = self.base / "logs"
        self.frames = self.base / "frames"
        self.max_buffer = max_buffer
        self.save_frames = save_frames
        self.frame_save_interval = max(1, frame_save_interval)
        self.img_width = img_width
        self.max_bytes = max_bytes

        for p in (self.base, self.logs, self.frames):
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        self._buffer: Deque[Dict[str, Any]] = deque()
        self._lock = threading.Lock()
        self._n_records = 0
        self._enforce_cap()

    def record(self, frame_idx: int, frame: Optional[np.ndarray],
               detections: List[Dict], agents: List[Dict], metadata: Dict) -> None:
        ts = round(time.time(), 3)
        item: Dict[str, Any] = {
            "frame_idx": int(frame_idx),
            "timestamp": ts,
            "detections": [
                {"id": int(d.get("id", -1)),
                 "cx": round(float(d.get("cx", 0)), 1),
                 "cy": round(float(d.get("cy", 0)), 1),
                 # Feet and box height, for ground calibration on replay
                 **({"fy": round(float(d["fy"]), 1), "bh": round(float(d["bh"]), 1)}
                    if d.get("bh") else {}),
                 "conf": round(float(d.get("confidence", d.get("conf", 0))), 3)}
                for d in detections
            ],
            "agents": [
                {"id": int(a.id), "x": round(float(a.x), 1), "y": round(float(a.y), 1),
                 "conf": round(float(a.confidence), 3)}
                for a in agents
            ],
            "meta": {
                "n_agents": len(agents),
                "fused_count": metadata.get("fused_count", None),
                "risk_label": metadata.get("risk_label", None),
                # Needed to replay the log through the twin (main.py --replay)
                "frame_size": metadata.get("frame_size"),
                "fps": metadata.get("source_fps"),
            },
            "frame_path": None,
        }

        # Save every Nth record's frame (resized) to disk if requested
        self._n_records += 1
        if self.save_frames and frame is not None and (self._n_records % self.frame_save_interval == 0):
            try:
                h, w = frame.shape[:2]
                scale = min(1.0, float(self.img_width) / max(1, w))
                if scale < 1.0:
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    small = cv2.resize(frame, (new_w, new_h))
                else:
                    small = frame
                fname = f"frame_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{frame_idx}.jpg"
                fpath = self.frames / fname
                cv2.imwrite(str(fpath), small, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                item["frame_path"] = str(fpath.as_posix())
            except Exception:
                item["frame_path"] = None

        with self._lock:
            self._buffer.append(item)
            # Keep memory bounded
            while len(self._buffer) > self.max_buffer:
                self._buffer.popleft()

    def buffer_len(self) -> int:
        with self._lock:
            return len(self._buffer)

    def flush(self) -> str:
        """Write buffered items to a JSONL file and clear the buffer.
        Returns path to the written file (str).
        """
        with self._lock:
            if not self._buffer:
                return ""
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            out = self.logs / f"experience_{ts}.jsonl"
            try:
                with open(out, "a", encoding="utf8") as fh:
                    while self._buffer:
                        it = self._buffer.popleft()
                        fh.write(json.dumps(it, default=str) + "\n")
            except Exception:
                # Requeue on failure
                return ""
        self._enforce_cap()
        return str(out.as_posix())

    def _enforce_cap(self) -> None:
        """Delete the oldest logs and frames once the buffer folder passes
        max_bytes, trimming to 90% so it doesn't prune on every flush."""
        try:
            files = [(e.stat().st_mtime, e.stat().st_size, e.path)
                     for d in (self.logs, self.frames) for e in os.scandir(d) if e.is_file()]
        except OSError:
            return
        total = sum(size for _, size, _ in files)
        if total <= self.max_bytes:
            return
        target = 0.9 * self.max_bytes
        for _, size, path in sorted(files):
            if total <= target:
                break
            try:
                os.remove(path)
                total -= size
            except OSError:
                pass

    def inspect(self, n: int = 10) -> List[Dict]:
        with self._lock:
            return list(self._buffer)[-n:]
