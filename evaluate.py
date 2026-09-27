"""
Layer N — Quantitative evaluation of the Crowd Digital Twin.

  python evaluate.py bench --source videos/demo.mp4 [--frames 200]
      End-to-end throughput and latency of the full pipeline (capture →
      payload) on this machine, next to the detector-only time.
  python evaluate.py mot17 --seq <MOT17>/train/MOT17-09-FRCNN [--frames N]
      Crowd-count MAE / RMSE / Pearson r and CLEAR-MOT tracking accuracy (MOTA).
  python evaluate.py ucf --root <UCF_CC_50 folder>
      Crowd-count error on UCF-CC-50 for body, head and body+head counting.
  python evaluate.py ade [--root dataset/eth_ucy] [--obs 8 --pred 12]
      Trajectory ADE / FDE (metres) of the constant-velocity baseline and the
      social force model used by the short-horizon simulation.

Every run prints its metrics and saves them to results/eval_<mode>_<time>.json.
"""

import argparse
import configparser
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

from main import (CDTPipeline, Config, SocialForceModel, fit_within,
                  fuse_fbox_hbox, load_eth_ucy)

WARMUP_FRAMES = 5


# ─── Metrics ──────────────────────────────────────────────────────────────────

def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) boxes in x1, y1, x2, y2 form."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter  = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def clear_mot(frames, iou_thr: float = 0.5) -> dict:
    """
    CLEAR-MOT metrics (Bernardin & Stiefelhagen, 2008). `frames` holds one
    (gt_ids, gt_boxes, pred_ids, pred_boxes) tuple per frame.
    """
    n_gt = fn = fp = idsw = matches = 0
    iou_sum = 0.0
    last = {}                                   # gt id → pred id it was last matched to
    for gt_ids, gt_boxes, pr_ids, pr_boxes in frames:
        n_gt += len(gt_ids)
        pairs = []
        if len(gt_ids) and len(pr_ids):
            iou = box_iou(gt_boxes, pr_boxes)
            col = {p: j for j, p in enumerate(pr_ids)}
            used_g, used_p = set(), set()
            # Keep last frame's correspondences while they still overlap
            for i, g in enumerate(gt_ids):
                j = col.get(last.get(g))
                if j is not None and j not in used_p and iou[i, j] >= iou_thr:
                    pairs.append((i, j))
                    used_g.add(i)
                    used_p.add(j)
            rest_g = [i for i in range(len(gt_ids)) if i not in used_g]
            rest_p = [j for j in range(len(pr_ids)) if j not in used_p]
            if rest_g and rest_p:
                sub  = iou[np.ix_(rest_g, rest_p)]
                cost = np.where(sub >= iou_thr, -sub, 1e6)
                for a, b in zip(*linear_sum_assignment(cost)):
                    if sub[a, b] >= iou_thr:
                        pairs.append((rest_g[a], rest_p[b]))
            for i, j in pairs:
                g, p = gt_ids[i], pr_ids[j]
                if g in last and last[g] != p:
                    idsw += 1
                last[g] = p
                iou_sum += iou[i, j]
        matches += len(pairs)
        fn += len(gt_ids) - len(pairs)
        fp += len(pr_ids) - len(pairs)
    return {
        "gt_boxes":        n_gt,
        "matches":         matches,
        "false_negatives": fn,
        "false_positives": fp,
        "id_switches":     idsw,
        "mota":            1 - (fn + fp + idsw) / max(n_gt, 1),
        "motp_iou":        iou_sum / max(matches, 1),
        "precision":       matches / max(matches + fp, 1),
        "recall":          matches / max(n_gt, 1),
    }


def count_errors(pred, gt) -> dict:
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    r = pearsonr(pred, gt)[0] if pred.std() > 0 and gt.std() > 0 else float("nan")
    return {
        "count_mae":       float(np.mean(np.abs(pred - gt))),
        "count_rmse":      float(np.sqrt(np.mean((pred - gt) ** 2))),
        "count_pearson_r": float(r),
    }


# ─── Modes ────────────────────────────────────────────────────────────────────

def bench(args) -> dict:
    """End-to-end throughput and latency of the full pipeline on a video file."""
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {args.source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or Config.SOURCE_FPS
    h, w = fit_within(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      Config.FRAME_HEIGHT, Config.FRAME_WIDTH)
    pipe = CDTPipeline(record_experience=False)
    pipe.configure(h, w, src_fps)

    e2e, detect = [], []
    idx = processed = 0
    while processed < args.frames + WARMUP_FRAMES:
        t_capture = time.time()
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % Config.FRAME_SKIP:
            continue
        payload = pipe.process_frame(frame, captured_at=t_capture, frame_idx=idx)
        json.dumps(payload)                     # serialisation for the WebSocket broadcast
        processed += 1
        if processed > WARMUP_FRAMES:
            e2e.append((time.time() - t_capture) * 1000)
            detect.append(payload["timing_ms"]["detect"])
    cap.release()
    if not e2e:
        raise SystemExit("Video too short to benchmark.")

    e2e, detect = np.array(e2e), np.array(detect)
    fps = 1000.0 / e2e.mean()
    return {
        "source":     args.source,
        "frames":     len(e2e),
        "resolution": f"{w}x{h}",
        "device":     pipe.device,
        "fp16":       pipe.half,
        "model":      pipe.model_name,
        # Frames are processed one after another here, so throughput = 1 / latency.
        # The live server captures in a separate thread and can only be faster.
        "end_to_end_fps": fps,
        "end_to_end_latency_ms": {
            "mean": e2e.mean(),
            "p50":  np.percentile(e2e, 50),
            "p95":  np.percentile(e2e, 95),
        },
        "detectors_ms":     detect.mean(),        # body + head model calls only
        "detectors_fps":    1000.0 / detect.mean(),
        "nfr_fps_pass":     fps >= 10,
        "nfr_latency_pass": np.percentile(e2e, 95) < 2000,
    }


def mot17(args) -> dict:
    """Count accuracy and CLEAR-MOT tracking accuracy on one MOT17 sequence."""
    seq  = Path(args.seq)
    info = configparser.ConfigParser()
    if not info.read(seq / "seqinfo.ini"):
        raise SystemExit(f"No seqinfo.ini in {seq}")
    s        = info["Sequence"]
    fps      = float(s.get("frameRate", Config.SOURCE_FPS))
    w, h     = int(s["imWidth"]), int(s["imHeight"])
    img_dir  = seq / s.get("imDir", "img1")
    ext      = s.get("imExt", ".jpg")
    n_frames = int(s["seqLength"])
    if args.frames:
        n_frames = min(n_frames, args.frames)

    gt = np.loadtxt(seq / "gt" / "gt.txt", delimiter=",", ndmin=2)
    gt = gt[(gt[:, 6] == 1) & (gt[:, 7] == 1)]          # scored pedestrians only
    gt_by_frame = defaultdict(list)
    for row in gt:
        gt_by_frame[int(row[0])].append(
            (int(row[1]), row[2], row[3], row[2] + row[4], row[3] + row[5]))

    Config.FRAME_SKIP = 1                               # MOT scoring needs every frame
    pipe = CDTPipeline(record_experience=False)
    pipe.configure(h, w, fps)

    frames, pred_counts, gt_counts = [], [], []
    for f in range(1, n_frames + 1):
        img = cv2.imread(str(img_dir / f"{f:06d}{ext}"))
        if img is None:
            break
        boxes = pipe.process_frame(img, frame_idx=f)["bounding_boxes"]
        g = gt_by_frame.get(f, [])
        frames.append((
            [x[0] for x in g],
            np.array([x[1:] for x in g], dtype=float).reshape(-1, 4),
            [b["id"] for b in boxes],
            np.array([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes],
                     dtype=float).reshape(-1, 4),
        ))
        pred_counts.append(len(boxes))
        gt_counts.append(len(g))
    if not frames:
        raise SystemExit(f"No frames found in {img_dir}")

    return {
        "sequence": seq.name,
        "frames":   len(frames),
        "device":   pipe.device,
        "model":    pipe.model_name,
        **count_errors(pred_counts, gt_counts),
        **clear_mot(frames),
    }


def ucf(args) -> dict:
    """Crowd-count error on UCF-CC-50 (head points in <n>_ann.mat)."""
    from scipy.io import loadmat

    root = Path(args.root)
    pipe = CDTPipeline(record_experience=False)
    rows = []
    for img_path in sorted(root.glob("*.jpg"), key=lambda p: (len(p.stem), p.stem)):
        ann = root / f"{img_path.stem}_ann.mat"
        img = cv2.imread(str(img_path))
        if not ann.exists() or img is None:
            continue
        gt    = len(loadmat(str(ann))["annPoints"])
        body  = pipe.detect(pipe.fbox_model, img, Config.YOLO_CONF, Config.YOLO_IMGSZ)
        heads = pipe.detect(pipe.hbox_model, img, Config.HBOX_CONF, Config.HBOX_IMGSZ)
        rows.append({"image": img_path.name, "gt": gt, "body": len(body),
                     "head": len(heads), "body_plus_head": len(fuse_fbox_hbox(body, heads))})
    if not rows:
        raise SystemExit(f"No UCF-CC-50 images with annotations in {root}")

    gt  = np.array([r["gt"] for r in rows], float)
    res = {"images": len(rows), "device": pipe.device, "max_det": Config.MAX_DET}
    for method in ("body", "head", "body_plus_head"):
        pred = np.array([r[method] for r in rows], float)
        res[f"{method}_mae"]  = np.mean(np.abs(pred - gt))
        res[f"{method}_rmse"] = np.sqrt(np.mean((pred - gt) ** 2))
    res["per_image"] = rows
    return res


def ade(args) -> dict:
    """ADE / FDE (metres) of the constant-velocity baseline and the social force model."""
    sfm = SocialForceModel()
    errors = {"linear": ([], []), "social_force": ([], [])}     # (ADE list, FDE list)
    n_files = 0
    for path in sorted(Path(args.root).glob("**/*.txt")):
        data = load_eth_ucy(path)
        if data is None:
            continue
        n_files += 1
        tracks   = defaultdict(dict)                 # pid → {frame: position}
        by_frame = defaultdict(list)                 # frame → pids
        for fr, pid, x, y in data:
            tracks[int(pid)][int(fr)] = np.array([x, y])
            by_frame[int(fr)].append(int(pid))
        diffs = [np.diff(sorted(t)) for t in tracks.values() if len(t) > 1]
        if not diffs:
            continue
        step = int(np.median(np.concatenate(diffs)))

        for f_o in sorted(by_frame):
            # Everyone with a velocity at the last observed frame takes part
            present = [pid for pid in by_frame[f_o] if f_o - step in tracks[pid]]
            targets = [pid for pid in present
                       if all(f_o - k * step in tracks[pid] for k in range(args.obs))
                       and all(f_o + k * step in tracks[pid] for k in range(1, args.pred + 1))]
            if not targets:
                continue
            pos = np.array([tracks[pid][f_o] for pid in present])
            vel = np.array([tracks[pid][f_o] - tracks[pid][f_o - step]
                            for pid in present]) / args.step_s
            sim   = np.array(sfm.simulate(pos, vel, args.pred, args.step_s))   # (pred, N, 2)
            ahead = np.arange(1, args.pred + 1)[:, None] * args.step_s
            for pid in targets:
                k     = present.index(pid)
                truth = np.array([tracks[pid][f_o + i * step] for i in range(1, args.pred + 1)])
                for name, pred in (("linear", pos[k] + ahead * vel[k]),
                                   ("social_force", sim[:, k])):
                    d = np.linalg.norm(pred - truth, axis=1)
                    errors[name][0].append(d.mean())
                    errors[name][1].append(d[-1])
    if not errors["linear"][0]:
        raise SystemExit(f"No ETH/UCY trajectory windows found in {args.root}")

    res = {"files": n_files, "windows": len(errors["linear"][0]),
           "obs": args.obs, "pred": args.pred, "step_s": args.step_s}
    for name, (a, f) in errors.items():
        res[f"{name}_ade_m"] = np.mean(a)
        res[f"{name}_fde_m"] = np.mean(f)
    return res


# ─── Entry Point ──────────────────────────────────────────────────────────────

def _clean(o):
    """JSON-safe results: numpy scalars → Python, NaN → None, floats rounded."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if np.isnan(o) else round(float(o), 4)
    return o


def main():
    parser = argparse.ArgumentParser(description="Crowd Digital Twin — Layer N evaluation")
    sub = parser.add_subparsers(dest="mode", required=True)

    b = sub.add_parser("bench", help="end-to-end FPS / latency on a video")
    b.add_argument("--source", default="videos/demo.mp4")
    b.add_argument("--frames", type=int, default=200)

    m = sub.add_parser("mot17", help="count error + MOTA on a MOT17 sequence")
    m.add_argument("--seq", required=True, help="e.g. MOT17/train/MOT17-09-FRCNN")
    m.add_argument("--frames", type=int, default=0, help="0 = whole sequence")

    u = sub.add_parser("ucf", help="count error on UCF-CC-50")
    u.add_argument("--root", required=True)

    a = sub.add_parser("ade", help="trajectory ADE/FDE on ETH/UCY")
    a.add_argument("--root", default=Config.ETH_UCY_PATH)
    a.add_argument("--obs", type=int, default=8)
    a.add_argument("--pred", type=int, default=12)
    a.add_argument("--step-s", type=float, default=0.4, help="seconds between samples")

    for p in (b, m, u):
        p.add_argument("--device", default=Config.DEVICE, help="auto | cpu | cuda:0")

    args = parser.parse_args()
    if getattr(args, "device", None):
        Config.DEVICE = args.device

    result = {"bench": bench, "mot17": mot17, "ucf": ucf, "ade": ade}[args.mode](args)
    result = _clean({"mode": args.mode,
                     "timestamp": datetime.now().isoformat(timespec="seconds"),
                     **result})

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"eval_{args.mode}_{datetime.now():%Y%m%dT%H%M%S}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in result.items() if k != "per_image"}, indent=2))
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
