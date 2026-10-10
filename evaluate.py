"""
Layer N — Quantitative evaluation of the Crowd Digital Twin.

  python evaluate.py bench --source videos/demo.mp4 [--frames 200]
      End-to-end throughput and latency of the full pipeline (capture →
      payload) on this machine, next to the detector-only time.
  python evaluate.py tracks --source videos/demo.mp4 --out dataset/tracks/demo
      One pass of the detector + tracker over a video, saved as MOT tracks
      (for `forecast` and `--replay`).
  python evaluate.py mot17 --seq <MOT17>/train/MOT17-09-FRCNN [--frames N] [--start-frac 0.85]
      Crowd-count MAE / RMSE / Pearson r and CLEAR-MOT tracking accuracy (MOTA).
  python evaluate.py dense --data crowdhuman:dataset/CrowdHuman [--max 500]
      Crowd-count error on CrowdHuman val: bodies, bodies + heads (normal
      and dense-mode resolution) and the head-point model.
  python evaluate.py forecast --tracks <MOT17 seq | tracks folder> ...
      The 20 s forecast against what actually happened: people's positions
      and zone counts at 2.4–20 s, for persistence, constant velocity, the
      old twin and the new twin (with ablations).

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

from forecast import (CrowdForecaster, Floor, ForecastParams, ForecastSkill, MotionFilter,
                      SceneMemory, walking_weight)
from main import (CDTPipeline, Config, GroundPlane, PedestrianCalibrator, SocialForceModel,
                  TrackReplay, ZoneManager, fit_within, fuse_fbox_hbox,
                  measurement_noise)

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
    pipe = CDTPipeline(record_experience=False, name=Path(args.source).stem)
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


def tracks(args) -> dict:
    """
    One pass of the live detector + tracker over a video file (no looping),
    saved as MOT-format tracker output (<out>/tracks.txt + seqinfo.ini) that
    `forecast` and `--replay` read. Frames are processed like the live
    pipeline: every FRAME_SKIP-th frame.
    """
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {args.source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or Config.SOURCE_FPS
    n_src   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = fit_within(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      Config.FRAME_HEIGHT, Config.FRAME_WIDTH)
    pipe = CDTPipeline(record_experience=False, name=Path(args.source).stem)
    pipe.configure(h, w, src_fps)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, idx, t0 = [], 0, time.perf_counter()
    last = None
    while True:
        ok, frame = cap.read()
        if not ok or (args.frames and idx >= args.frames):
            break
        idx += 1
        if (idx - 1) % Config.FRAME_SKIP:
            continue
        gap  = 1.0 if last is None else (idx - last) / Config.FRAME_SKIP
        last = idx
        payload = pipe.process_frame(frame, frame_gap=gap, frame_idx=idx)
        for b in payload["bounding_boxes"]:
            rows.append((idx, b["id"], b["x1"], b["y1"], b["x2"] - b["x1"], b["y2"] - b["y1"],
                         b["conf"], -1, -1, -1))
        if idx % 50 < Config.FRAME_SKIP:
            print(f"  frame {idx}/{n_src}: {len(payload['bounding_boxes'])} people "
                  f"({(time.perf_counter() - t0) / max(1, idx // Config.FRAME_SKIP):.2f} s/frame)")
    cap.release()
    np.savetxt(out / "tracks.txt", np.array(rows, float).reshape(-1, 10), delimiter=",",
               fmt=["%d", "%d", "%.1f", "%.1f", "%.1f", "%.1f", "%.3f", "%d", "%d", "%d"])
    (out / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={Path(args.source).stem}\nimDir=img1\nframeRate={src_fps:g}\n"
        f"seqLength={idx}\nimWidth={w}\nimHeight={h}\nimExt=.jpg\n", encoding="utf-8")
    ids = {r[1] for r in rows}
    return {"source": args.source, "frames": idx, "processed": idx // Config.FRAME_SKIP,
            "resolution": f"{w}x{h}", "fps": src_fps, "tracks": len(ids),
            "boxes": len(rows), "out": (out / "tracks.txt").as_posix()}


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
    # Frames the body model never trained on: its notebook (reference/yolo26smodel.ipynb)
    # kept the last int(N × 0.15) frames of each MOT17 train sequence for validation
    n_held = int(n_frames * round(1.0 - args.start_frac, 6))
    first  = n_frames - n_held + 1 if args.start_frac > 0 else 1
    if args.frames:
        n_frames = min(n_frames, first + args.frames - 1)

    gt = np.loadtxt(seq / "gt" / "gt.txt", delimiter=",", ndmin=2)
    gt = gt[(gt[:, 6] == 1) & (gt[:, 7] == 1)]          # scored pedestrians only
    gt_by_frame = defaultdict(list)
    for row in gt:
        gt_by_frame[int(row[0])].append(
            (int(row[1]), row[2], row[3], row[2] + row[4], row[3] + row[5]))

    Config.FRAME_SKIP = 1                               # MOT scoring needs every frame
    pipe = CDTPipeline(record_experience=False, name=seq.name)
    pipe.configure(h, w, fps)

    frames, pred_counts, gt_counts = [], [], []
    for f in range(first, n_frames + 1):
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
        "first_frame": first,
        "held_out": args.start_frac > 0,
        "device":   pipe.device,
        "model":    pipe.model_name,
        **count_errors(pred_counts, gt_counts),
        **clear_mot(frames),
    }


def dense(args) -> dict:
    """
    Crowd-count error on CrowdHuman val (or your own `points` folders — see
    train_dense.py) for: bodies; bodies + heads at normal and at dense-mode
    resolution (what the twin counts); and the dense-crowd point model, when
    its weights are given or in place.
    """
    import random
    from train_dense import list_samples
    samples = []
    for spec in args.data:
        kind, root = spec.split(":", 1)
        samples += [(kind, *smp) for smp in list_samples(kind, root.partition("@")[0], "test")]
    if not samples:
        raise SystemExit("No test images found.")
    if args.max and len(samples) > args.max:
        samples = random.Random(0).sample(samples, args.max)
    pipe = CDTPipeline(record_experience=False)
    points_model = pipe.points_model
    if args.weights or (points_model is not None and points_model.enhance != args.enhance):
        from dense import PointCounter
        points_model = PointCounter(args.weights or Config.DENSE_MODEL, pipe.device,
                                    Config.DENSE_THRESHOLD, half=pipe.half, enhance=args.enhance)
    rows = []
    from train_dense import load_image
    for i, (kind, path, pts, ignore) in enumerate(samples):
        try:                          # unlabelled regions blanked, like in training
            img, pts = load_image(path, pts, args.max_side, ignore)
        except RuntimeError:
            continue
        body  = pipe.detect(pipe.fbox_model, img, Config.YOLO_CONF, Config.YOLO_IMGSZ)
        heads = pipe.detect(pipe.hbox_model, img, Config.HBOX_CONF, Config.HBOX_IMGSZ)
        heads_d = pipe.detect(pipe.hbox_model, img, Config.HBOX_CONF_DENSE, Config.HBOX_IMGSZ_DENSE)
        row = {"dataset": kind, "image": Path(path).name, "gt": len(pts), "body": len(body),
               "body_plus_head": len(fuse_fbox_hbox(body, heads)),
               "body_plus_head_dense": len(fuse_fbox_hbox(body, heads_d))}
        if points_model is not None:
            row["points"] = len(points_model(img)[0])
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(samples)} images")
    methods = [m for m in ("body", "body_plus_head", "body_plus_head_dense", "points") if m in rows[0]]
    res = {"images": len(rows), "device": pipe.device, "enhance": args.enhance,
           "point_model": (args.weights or (Config.DENSE_MODEL if points_model else None)),
           "mean_gt": float(np.mean([r["gt"] for r in rows]))}
    for kind in sorted({r["dataset"] for r in rows}) + ["all"]:
        sel = [r for r in rows if kind == "all" or r["dataset"] == kind]
        gt = np.array([r["gt"] for r in sel], float)
        res[kind] = {m: {"mae": float(np.mean(np.abs(np.array([r[m] for r in sel]) - gt))),
                         "rmse": float(np.sqrt(np.mean((np.array([r[m] for r in sel]) - gt) ** 2))),
                         "recall_of_count": float(np.sum([r[m] for r in sel]) / max(gt.sum(), 1))}
                     for m in methods}
    res["per_image"] = rows
    return res


# ─── Forecast vs reality ──────────────────────────────────────────────────────

class TrackSource:
    """
    Tracks on the ground for the forecast evaluation: per processed frame,
    the time, track ids, positions (metres) and their measurement noise,
    plus the floor and a zone lookup.
    """
    def __init__(self, name, frames, floor: Floor, zone_of, n_zones: int, frame_s: float):
        self.name, self.frames, self.floor = name, frames, floor
        self.zone_of, self.n_zones, self.frame_s = zone_of, n_zones, frame_s

    @classmethod
    def from_image_tracks(cls, path, calibration=None) -> "TrackSource":
        """MOT-format tracks (MOT ground truth or `evaluate.py tracks` output), in pixels."""
        rep = TrackReplay(str(path))
        h, w = rep.size
        raw = []
        for fr, dets in rep.frames():
            for d in dets:
                d["fx"], d["fy"], d["bh"] = d["cx"], d["y2"], d["y2"] - d["y1"]
            raw.append((fr, dets))
        raw = [(fr, d) for fr, d in raw if d]
        if not raw:
            raise SystemExit(f"No tracks in {path}")
        first = raw[0][0]
        raw = [(fr, d) for fr, d in raw if (fr - first) % Config.FRAME_SKIP == 0]
        # Ground: a calibration file, else estimated from the boxes (as the live twin does)
        if calibration:
            ground, why = GroundPlane.from_file(calibration, h, w), f"from {calibration}"
        else:
            cal = PedestrianCalibrator(h, w)
            for _, dets in raw:
                cal.add(dets)
            ground, why = cal.fit()
            if ground is None:
                ground, why = GroundPlane.flat(h, w), f"flat {Config.SCENE_WIDTH_M:g} m wide ({why})"
        feet_rows = [d["fy"] for _, dets in raw for d in dets if d["confidence"] >= 0.4]
        if len(feet_rows) >= Config.FLOOR_MIN_FEET:
            top = max(0.0, float(np.percentile(feet_rows, 1)) - 0.02 * h)
            ground.floor_top = top if top > 0 else None
        zm = ZoneManager(h, w, ground)
        floor = Floor(ground.boundary())
        frames = []
        for fr, dets in raw:
            feet = [[d["fx"], d["fy"]] for d in dets]
            frames.append((fr / rep.fps, [d["id"] for d in dets], ground.to_world(feet),
                           measurement_noise(ground, feet, [d["bh"] for d in dets])))

        def zone_of(pts):
            pts = np.asarray(pts, float).reshape(-1, 2)
            z = zm.index_many(ground.to_image(pts))
            return np.where(floor.depth(pts).min(axis=1) >= -1e-6, z, -1)

        pp = Path(path)
        # MOT17 ground truth is <SEQ>/gt/gt.txt: name it after the sequence, not "gt"
        # (three sequences called "gt" overwrote each other in the results)
        name = (pp.parent.parent.name if pp.parent.name == "gt" else pp.parent.name)             if pp.name in ("tracks.txt", "gt.txt") else pp.stem
        src = cls(name, frames, floor, zone_of, len(zm.zones),
                  Config.FRAME_SKIP / rep.fps)
        src.ground_note = why
        return src

class _OldVelocity:
    """The twin's former velocity estimate: 1 s least-squares slope, EMA 0.4 (for the old baseline)."""
    def __init__(self):
        self.track, self.v = defaultdict(list), defaultdict(lambda: np.zeros(2))

    def step(self, t, ids, world):
        out = np.zeros((len(ids), 2))
        for i, (k, p) in enumerate(zip(ids, world)):
            tr = self.track[k]
            tr.append((t, p[0], p[1]))
            while tr[0][0] < t - 1.0:
                tr.pop(0)
            if len(tr) > 1:
                a = np.array(tr)
                tm = a[:, 0] - a[:, 0].mean()
                if (tm ** 2).sum() > 0:
                    slope = (tm[:, None] * a[:, 1:]).sum(axis=0) / (tm ** 2).sum()
                    self.v[k] = 0.4 * slope + 0.6 * self.v[k]
            out[i] = self.v[k]
        return out


FORECAST_MODELS = ("persistence", "constant_velocity", "old_twin", "new_twin",
                   "new_uncalibrated", "new_no_flow", "new_no_arrivals",
                   "new_no_walk_gate", "new_no_avoidance", "new_no_weidmann")


def _ablated(params: ForecastParams) -> dict:
    """The forecast with one component switched off at a time."""
    from dataclasses import replace
    return {
        # everyone who moves at all walks (no speed / significance gate)
        "new_no_walk_gate": replace(params, walk_min_speed=0.0, walk_ramp=1e-6,
                                    walk_z_lo=-2.0, walk_z_hi=-1.0),
        # nobody steps aside for people they are about to meet
        "new_no_avoidance": replace(params, ttc_k=0.0),
        # walking speed does not drop with local density
        "new_no_weidmann": replace(params, weidmann_gamma=1e6),
    }


def _zone_counts(src: TrackSource, paths: np.ndarray, active=None) -> np.ndarray:
    """(M, S, 2) paths → (S, Z) people per zone at each step."""
    m, s = paths.shape[:2]
    zi = src.zone_of(paths.reshape(-1, 2)).reshape(m, s)
    if active is not None:
        zi = np.where(active, zi, -1)
    return np.stack([np.bincount(zi[:, k][zi[:, k] >= 0], minlength=src.n_zones)
                     for k in range(s)]).astype(float)


def _forecast_source(src: TrackSource, params: ForecastParams, every_s: float,
                     warmup_s: float, horizons) -> dict:
    """
    Replay the tracks in time order like the live twin: every `every_s`, run
    each model from the current state (the new twin calibrates its zone
    counts with ForecastSkill, from its own earlier forecasts only), then
    score everything against what happened.
    """
    mf, old = MotionFilter(), _OldVelocity()
    mem = SceneMemory(src.floor)
    skill = ForecastSkill(params.steps, params.step_s)
    sfm = SocialForceModel()
    fc = CrowdForecaster(params)
    ablations = {k: CrowdForecaster(p) for k, p in _ablated(params).items()}
    poly = src.floor.vertices if src.floor.valid else None
    steps = params.steps
    ahead = np.arange(steps + 1)[None, :, None] * params.step_s
    last_seen = {}
    truth = []                                   # (t, {id: raw world position}, zone counts)
    snaps = []
    t_first = src.frames[0][0]
    next_eval = t_first + warmup_s
    for t, ids, world, R in src.frames:
        if ids:
            pos, vel, std = mf.step(t, ids, world, R)
            walk = walking_weight(np.linalg.norm(vel, axis=1), std)
        else:
            pos = vel = np.zeros((0, 2))
            std = walk = np.zeros(0)
        mem.observe(t, ids, pos, vel, walk)
        v_old = old.step(t, ids, world)
        for k in ids:
            last_seen[k] = t
        stale = [k for k, ts in last_seen.items() if t - ts > Config.TRACKER_BUFFER * src.frame_s]
        mf.drop(stale)
        for k in stale:
            del last_seen[k]
        world = np.asarray(world, float).reshape(-1, 2)
        zc = _zone_counts(src, world[:, None])[0] if len(world) else np.zeros(src.n_zones)
        truth.append((t, dict(zip(ids, world)), zc))
        skill.observe(t, zc, tol=0.6 * src.frame_s)
        if t >= next_eval and ids:
            runs = {
                "persistence":       (np.repeat(world[:, None], steps + 1, axis=1), None),
                "constant_velocity": (pos[:, None] + ahead * vel[:, None], None),
                "old_twin": (np.stack([world] + sfm.simulate(world, v_old, steps, params.step_s,
                                                             bounds=poly), axis=1), None),
            }
            flow, arrivals = mem.flow_field(), mem.arrivals(t)
            for name, kw in (("new_uncalibrated", {}), ("new_no_flow", {"flow": None}),
                             ("new_no_arrivals", {"arrivals": (0.0, [])})):
                args = dict(floor=src.floor, flow=flow, arrivals=arrivals)
                args.update(kw)
                r = fc.run(pos, vel, std, **args)
                runs[name] = (r.paths, _zone_counts(src, r.paths, r.active()))
            for name, f in ablations.items():
                r = f.run(pos, vel, std, floor=src.floor, flow=flow, arrivals=arrivals)
                runs[name] = (r.paths, _zone_counts(src, r.paths, r.active()))
            raw = runs["new_uncalibrated"][1]
            runs["new_twin"] = (runs["new_uncalibrated"][0], skill.calibrate(zc, raw))
            skill.add(t, zc, raw)
            snaps.append(dict(t=t, ids=list(ids), raw=world, runs=runs))
            next_eval = t + every_s

    times = np.array([tr[0] for tr in truth])
    tol = 0.6 * src.frame_s
    stats = {m: {h: defaultdict(list) for h in horizons} for m in FORECAST_MODELS}
    for s in snaps:
        runs = s["runs"]
        for h in horizons:
            k = int(round(h / params.step_s))
            j = int(np.argmin(np.abs(times - (s["t"] + k * params.step_s))))
            if abs(times[j] - (s["t"] + k * params.step_s)) > tol:
                continue                                    # past the end of the recording
            t_true, pos_true, zc_true = truth[j]
            have = [i for i, pid in enumerate(s["ids"]) if pid in pos_true]
            true_xy = np.array([pos_true[s["ids"][i]] for i in have]).reshape(-1, 2)
            still = np.linalg.norm(true_xy - s["raw"][have], axis=1) < 0.3 if have else np.zeros(0, bool)
            # Average error along the way (ids seen at each intermediate step)
            mids = [int(np.argmin(np.abs(times - (s["t"] + q * params.step_s))))
                    for q in range(1, k + 1)]
            for name, (paths, counts) in runs.items():
                zc = counts[k] if counts is not None else _zone_counts(src, paths[:, k:k + 1])[0]
                st = stats[name][h]
                st["zone_mae"].append(np.abs(zc - zc_true).mean())
                st["total_err"].append(abs(zc.sum() - zc_true.sum()))
                st["true_total"].append(zc_true.sum())
                if have:
                    err = np.linalg.norm(paths[have, k] - true_xy, axis=1)
                    st["fde"].extend(err.tolist())
                    if still.any():
                        st["standing_drift"].extend(np.linalg.norm(
                            paths[have, k][still] - paths[have, 0][still], axis=1).tolist())
                    along = []
                    for q, jq in enumerate(mids, start=1):
                        pt = truth[jq][1]
                        e = [np.linalg.norm(paths[i, q] - pt[s["ids"][i]])
                             for i in have if s["ids"][i] in pt]
                        if e:
                            along.append(np.mean(e))
                    if along:
                        st["ade"].append(float(np.mean(along)))
    out = {"frames": len(src.frames), "duration_s": round(src.frames[-1][0] - t_first, 1),
           "snapshots": len(snaps), "ground": getattr(src, "ground_note", ""),
           "mean_people": float(np.mean([len(f[1]) for f in src.frames])), "models": {}}
    for name in FORECAST_MODELS:
        out["models"][name] = {}
        for h in horizons:
            st = stats[name][h]
            if not st["zone_mae"]:
                continue
            out["models"][name][f"{h:g}s"] = {
                "windows":        len(st["zone_mae"]),
                "ade_m":          float(np.mean(st["ade"])) if st["ade"] else None,
                "fde_m":          float(np.mean(st["fde"])) if st["fde"] else None,
                "people_scored":  len(st["fde"]),
                "zone_count_mae": float(np.mean(st["zone_mae"])),
                "total_count_err": float(np.mean(st["total_err"])),
                "standing_drift_m": float(np.mean(st["standing_drift"])) if st["standing_drift"] else None,
            }
    return out


def forecast(args) -> dict:
    """
    Forecast vs reality: from snapshots of the tracked crowd, forecast each
    model up to 20 s ahead and compare with where the people actually were
    and how many were in each zone. Models: persistence (nobody moves),
    constant velocity, the old twin (1 s slope velocity + Helbing social
    force, reflecting at the view edges) and the new twin (forecast.py), plus
    the new twin without the learned flow / without arrivals.
    """
    params = ForecastParams(steps=Config.PRED_HORIZON, step_s=Config.SIM_STEP_S,
                            substeps=Config.SIM_SUBSTEPS)
    horizons = [h for h in args.horizons if h <= params.steps * params.step_s + 1e-9]
    sources = []
    for spec in args.tracks:
        p = Path(spec)
        if p.is_dir() and (p / "gt" / "gt.txt").exists():
            sources.append(TrackSource.from_image_tracks(p / "gt" / "gt.txt", args.calibration))
        elif p.is_dir() and (p / "tracks.txt").exists():
            sources.append(TrackSource.from_image_tracks(p / "tracks.txt", args.calibration))
        else:
            sources.append(TrackSource.from_image_tracks(p, args.calibration))
    res = {"horizons_s": horizons, "every_s": args.every, "warmup_s": args.warmup,
           "sources": {}}
    for src in sources:
        print(f"  {src.name}: {len(src.frames)} frames …")
        res["sources"][src.name] = _forecast_source(src, params, args.every, args.warmup, horizons)
    # Pooled over sources (weighted by windows)
    pooled = {}
    for name in FORECAST_MODELS:
        pooled[name] = {}
        for h in horizons:
            key = f"{h:g}s"
            rows = [s["models"][name][key] for s in res["sources"].values() if key in s["models"][name]]
            if not rows:
                continue
            w = np.array([r["windows"] for r in rows], float)
            wp = np.array([r["people_scored"] for r in rows], float)
            avg = lambda f, ww: (float(np.average([r[f] for r in rows if r[f] is not None],
                                                  weights=[x for r, x in zip(rows, ww) if r[f] is not None]))
                                 if any(r[f] is not None for r in rows) and
                                 sum(x for r, x in zip(rows, ww) if r[f] is not None) > 0 else None)
            pooled[name][key] = {"ade_m": avg("ade_m", wp), "fde_m": avg("fde_m", wp),
                                 "zone_count_mae": avg("zone_count_mae", w),
                                 "total_count_err": avg("total_count_err", w),
                                 "standing_drift_m": avg("standing_drift_m", wp)}
    res["pooled"] = pooled
    # Table
    print(f"\n{'model':<18}" + "".join(f"{f'{h:g}s':>22}" for h in horizons))
    print(f"{'':<18}" + "".join(f"{'FDE m / zone MAE':>22}" for _ in horizons))
    for name in FORECAST_MODELS:
        cells = []
        for h in horizons:
            r = pooled[name].get(f"{h:g}s")
            cells.append(f"{r['fde_m']:.2f} / {r['zone_count_mae']:.2f}" if r and r["fde_m"] is not None
                         else (f"– / {r['zone_count_mae']:.2f}" if r else "–"))
        print(f"{name:<18}" + "".join(f"{c:>22}" for c in cells))
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

    t = sub.add_parser("tracks", help="one pass of detector + tracker over a video → MOT tracks")
    t.add_argument("--source", required=True)
    t.add_argument("--out", required=True, help="folder for tracks.txt + seqinfo.ini")
    t.add_argument("--frames", type=int, default=0, help="0 = whole video")

    m = sub.add_parser("mot17", help="count error + MOTA on a MOT17 sequence")
    m.add_argument("--seq", required=True, help="e.g. MOT17/train/MOT17-09-FRCNN")
    m.add_argument("--frames", type=int, default=0, help="0 = whole sequence")
    m.add_argument("--start-frac", type=float, default=0.0,
                   help="score only frames after this fraction of the sequence "
                        "(0.85 = the body model's held-out validation frames)")

    d = sub.add_parser("dense", help="crowd-count error on CrowdHuman val")
    d.add_argument("--data", action="append", required=True,
                   help="kind:path as in train_dense.py (crowdhuman, jhu, points)")
    d.add_argument("--weights", help="point-model checkpoint (default: Config.DENSE_MODEL if present)")
    d.add_argument("--max", type=int, default=0, help="at most N images (0 = all)")
    d.add_argument("--max-side", type=int, default=2048, help="downscale larger images")
    d.add_argument("--enhance", action="store_true",
                   help="equalise contrast before the point model (as the live pipeline does)")

    f = sub.add_parser("forecast", help="20 s forecast vs what actually happened")
    f.add_argument("--tracks", nargs="+", required=True,
                   help="MOT17 sequence folders, `tracks` output folders or MOT CSVs")
    f.add_argument("--calibration", help="ground calibration JSON for pixel tracks "
                                          "(default: estimated from the boxes)")
    f.add_argument("--every", type=float, default=1.0, help="seconds between forecasts")
    f.add_argument("--warmup", type=float, default=3.0, help="seconds of history before the first")
    f.add_argument("--horizons", type=float, nargs="+", default=[2.4, 4.8, 8.0, 12.0, 16.0, 20.0])

    for p in (b, t, m, d):
        p.add_argument("--device", default=Config.DEVICE, help="auto | cpu | cuda:0")

    args = parser.parse_args()
    if getattr(args, "device", None):
        Config.DEVICE = args.device

    result = {"bench": bench, "tracks": tracks, "mot17": mot17,
              "forecast": forecast, "dense": dense}[args.mode](args)
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
