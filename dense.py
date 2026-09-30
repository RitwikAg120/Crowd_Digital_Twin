"""
Dense-crowd head localisation — P2PNet (Song et al., "Rethinking Counting and
Localization in Crowds: A Purely Point-Based Framework", ICCV 2021).

In a dense crowd (a ghat at the Kumbh, a stampede-risk corridor) heads are
3–10 px and bodies are hidden, so box detectors miss most people. P2PNet
predicts one point per head directly — trained here on CrowdHuman's head
annotations (train_dense.py) — and each point becomes a person in the twin
like an unmatched head from the head detector.

No weights ship with this repository: train them on the GB10
(scripts/run_gb10.sh) and put the file at Config.DENSE_MODEL; without it,
dense mode uses the head detector only.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:                       # the twin-only environment has no PyTorch
    torch = None
    nn = None


if nn is not None:

    def _vgg16_bn_features():
        """torchvision's VGG16-BN feature stack without downloading ImageNet weights."""
        import torchvision
        return torchvision.models.vgg16_bn(weights=None).features

    class Backbone(nn.Module):
        """VGG16-BN split after each pooling stage, as in P2PNet (strides 2, 4, 8, 16)."""
        def __init__(self, features=None):
            super().__init__()
            f = list((features if features is not None else _vgg16_bn_features()).children())
            self.body1 = nn.Sequential(*f[:13])
            self.body2 = nn.Sequential(*f[13:23])
            self.body3 = nn.Sequential(*f[23:33])
            self.body4 = nn.Sequential(*f[33:43])

        def forward(self, x):
            out = []
            for layer in (self.body1, self.body2, self.body3, self.body4):
                x = layer(x)
                out.append(x)
            return out

    class Decoder(nn.Module):
        """FPN-style decoder (P3 at stride 4, P4 at stride 8, P5 at stride 16)."""
        def __init__(self, c3=256, c4=512, c5=512, fs=256):
            super().__init__()
            self.P5_1 = nn.Conv2d(c5, fs, 1)
            self.P5_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
            self.P5_2 = nn.Conv2d(fs, fs, 3, padding=1)
            self.P4_1 = nn.Conv2d(c4, fs, 1)
            self.P4_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
            self.P4_2 = nn.Conv2d(fs, fs, 3, padding=1)
            self.P3_1 = nn.Conv2d(c3, fs, 1)
            self.P3_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
            self.P3_2 = nn.Conv2d(fs, fs, 3, padding=1)

        def forward(self, c3, c4, c5):
            p5 = self.P5_1(c5)
            p5_up = self.P5_upsampled(p5)
            p4 = self.P4_1(c4) + p5_up
            p4_up = self.P4_upsampled(p4)
            p4 = self.P4_2(p4)
            p3 = self.P3_2(self.P3_1(c3) + p4_up)
            return p3, p4, self.P5_2(p5)

    class _Head(nn.Module):
        """Regression / classification branch (conv3, conv4 exist in the checkpoints but are unused)."""
        def __init__(self, cin, cout, fs=256):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, fs, 3, padding=1)
            self.act1 = nn.ReLU()
            self.conv2 = nn.Conv2d(fs, fs, 3, padding=1)
            self.act2 = nn.ReLU()
            self.conv3 = nn.Conv2d(fs, fs, 3, padding=1)
            self.act3 = nn.ReLU()
            self.conv4 = nn.Conv2d(fs, fs, 3, padding=1)
            self.act4 = nn.ReLU()
            self.output = nn.Conv2d(fs, cout, 3, padding=1)

        def forward(self, x):
            return self.output(self.act2(self.conv2(self.act1(self.conv1(x)))))

    class P2PNet(nn.Module):
        """
        Four anchor points per stride-8 cell; each predicts an offset and a
        head / no-head score. forward() returns logits (B, K, 2) and points
        (B, K, 2) in input pixels (x, y).
        """
        STRIDE = 8

        def __init__(self, backbone_features=None, row: int = 2, line: int = 2):
            super().__init__()
            self.row, self.line = row, line
            a = row * line
            self.backbone = Backbone(backbone_features)
            self.fpn = Decoder()
            self.regression = _Head(256, a * 2)
            self.classification = _Head(256, a * 2)

        def anchors(self, h: int, w: int, device) -> "torch.Tensor":
            s = self.STRIDE
            gh, gw = (h + s - 1) // s, (w + s - 1) // s
            ay = ((torch.arange(1, self.row + 1, device=device) - 0.5) * s / self.row - s / 2)
            ax = ((torch.arange(1, self.line + 1, device=device) - 0.5) * s / self.line - s / 2)
            ay, ax = torch.meshgrid(ay, ax, indexing="ij")
            base = torch.stack([ax.reshape(-1), ay.reshape(-1)], dim=1)            # (A, 2)
            sy = (torch.arange(gh, device=device) + 0.5) * s
            sx = (torch.arange(gw, device=device) + 0.5) * s
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")
            shifts = torch.stack([sx.reshape(-1), sy.reshape(-1)], dim=1)          # (K, 2)
            return (shifts[:, None, :] + base[None, :, :]).reshape(-1, 2).float()

        def forward(self, x):
            f = self.backbone(x)
            _, p4, _ = self.fpn(f[1], f[2], f[3])
            b = x.shape[0]
            reg = self.regression(p4).permute(0, 2, 3, 1).reshape(b, -1, 2) * 100
            cls = self.classification(p4).permute(0, 2, 3, 1).reshape(b, -1, 2)
            pts = reg + self.anchors(x.shape[2], x.shape[3], x.device)[None]
            return cls, pts


def load_p2pnet(path, device: str = "cpu") -> "P2PNet":
    """A P2PNet from an official or train_dense.py checkpoint (weights only — no pickled code)."""
    if torch is None:
        raise RuntimeError("The dense-crowd model needs PyTorch (the full environment).")
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    row = int(ckpt.get("row", 2)) if isinstance(ckpt, dict) else 2
    model = P2PNet(row=row, line=row)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if [k for k in missing if not k.endswith("num_batches_tracked")] or unexpected:
        raise RuntimeError(f"{path} is not a P2PNet checkpoint "
                           f"(missing {missing[:3]}…, unexpected {unexpected[:3]}…)")
    return model.to(device).eval()


class PointCounter:
    """
    Heads in a frame as points: the image is scaled so its width is at least
    MIN_WIDTH (small, distant heads get enough pixels), padded to a multiple
    of 128, normalised like ImageNet, and every anchor with a head score ≥
    threshold is a head. Returns (N, 2) points in frame pixels and scores.
    """
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD  = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, weights, device: str = "cpu", threshold: float = 0.5,
                 min_width: int = 1280, max_side: int = 2560, half: bool = False):
        self.model = load_p2pnet(weights, device)
        self.device, self.threshold = device, threshold
        self.min_width, self.max_side = min_width, max_side
        self.half = half and str(device).startswith("cuda")
        if self.half:
            self.model = self.model.half()
        self.name = Path(weights).name

    def __call__(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        import cv2
        h, w = frame_bgr.shape[:2]
        s = max(1.0, self.min_width / w)
        s = min(s, self.max_side / max(h, w))
        img = cv2.resize(frame_bgr, (int(round(w * s)), int(round(h * s)))) if s != 1.0 else frame_bgr
        hh, ww = img.shape[:2]
        ph, pw = (128 - hh % 128) % 128, (128 - ww % 128) % 128
        x = (img[:, :, ::-1].astype(np.float32) / 255.0 - self.MEAN) / self.STD
        x = np.pad(x, ((0, ph), (0, pw), (0, 0)))
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(self.device)
        if self.half:
            t = t.half()
        with torch.inference_mode():
            logits, pts = self.model(t)
        score = logits.float().softmax(-1)[0, :, 1].cpu().numpy()
        pts = pts[0].float().cpu().numpy()
        keep = (score >= self.threshold) & (pts[:, 0] < ww) & (pts[:, 1] < hh) & \
               (pts[:, 0] >= 0) & (pts[:, 1] >= 0)
        return pts[keep] / s, score[keep]


def head_sizes(points: np.ndarray, k: int = 3, lo: float = 4.0, hi: float = 60.0) -> np.ndarray:
    """
    Head size (px) for points without boxes: in a dense crowd, heads are about
    as far apart as they are big, so use the mean distance to the k nearest
    neighbours (the usual geometry-adaptive kernel of crowd counting).
    """
    from scipy.spatial import cKDTree
    p = np.asarray(points, float).reshape(-1, 2)
    if len(p) < 2:
        return np.full(len(p), 16.0)
    d, _ = cKDTree(p).query(p, k=min(k + 1, len(p)))
    return np.clip(d[:, 1:].mean(axis=1) * 0.8, lo, hi)
