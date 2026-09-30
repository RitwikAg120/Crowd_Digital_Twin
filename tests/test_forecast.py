"""
Checks for the motion filter, scene memory, forecaster and self-calibration
(forecast.py) and their use in the twin. Needs no PyTorch:

    venv-twin\\Scripts\\python tests\\test_forecast.py
"""
import json

import numpy as np

import runner  # noqa: F401  (sets the working directory and import path)
from forecast import (CrowdForecaster, Floor, ForecastParams, ForecastSkill, MotionFilter,
                      SceneMemory, debiased_speed, walking_weight)
from main import (Config, GroundPlane, TwinPipeline, TwinService, measurement_noise)

H, W = 720, 1280

Config.AUTO_CALIBRATE = False       # the checks must not write calibration files into the project
CAM = GroundPlane.from_camera(H, W, 6.0, 25.0, 65.0)
P = ForecastParams()


def _box(w=20.0, h=12.0, kinds=("view", "view", "view", "view")):
    pts = [[0, 0], [w, 0], [w, h], [0, h]]
    return Floor([{"a": pts[k - 1], "b": pts[k], "kind": kinds[k - 1]} for k in range(4)])


def _filter_track(world_xy, t, bh=60.0, jitter_px=1.0, seed=0):
    """Feed one person's true ground path through the perspective + pixel jitter + filter."""
    rng = np.random.default_rng(seed)
    mf = MotionFilter()
    for ti, xy in zip(t, world_xy):
        px = CAM.to_image([xy])[0] + rng.normal(0, jitter_px, 2)
        R = measurement_noise(CAM, [px], [bh])
        pos, vel, std = mf.step(ti, [1], CAM.to_world([px]), R)
    return pos[0], vel[0], std[0]


def test_standing_far_away_reads_as_standing():
    t = np.arange(40) * 0.4                       # 16 s at the CPU pipeline's pace
    far = np.tile([[2.0, 40.0]], (40, 1))          # 40 m away: a pixel is ~0.5 m of depth
    pos, vel, std = _filter_track(far, t, bh=18, jitter_px=1.5)
    w = walking_weight(np.linalg.norm(vel), std)
    assert w == 0, f"walking weight {w:.2f} (speed {np.linalg.norm(vel):.2f} ± {std:.2f})"
    return f"far standing person: {np.linalg.norm(vel):.2f} ± {std:.2f} m/s → standing"


def test_walkers_read_true_speed_near_and_far():
    t = np.arange(40) * 0.4
    out = []
    for y0, bh in ((10.0, 90.0), (30.0, 25.0)):
        path = np.column_stack([-4 + 1.3 * t, np.full_like(t, y0)])
        pos, vel, std = _filter_track(path, t, bh=bh)
        s = np.linalg.norm(vel)
        assert abs(s - 1.3) < 0.3 and walking_weight(s, std) == 1, f"{y0} m: {s:.2f} m/s"
        out.append(f"{s:.2f}")
    return "1.3 m/s walkers read " + " / ".join(out) + " m/s (near / far)"


def test_debiased_speed():
    assert debiased_speed(0.42, 0.3) == 0.0
    assert abs(debiased_speed(1.3, 0.1) - np.sqrt(1.3 ** 2 - 0.02)) < 1e-9


def test_standing_crowd_stays_put():
    rng = np.random.default_rng(1)
    pos = rng.uniform([3, 3], [17, 9], (120, 2))  # dense (~1.4 p/m²), standing, noisy velocity
    vel = rng.normal(0, 0.15, (120, 2))
    r = CrowdForecaster().run(pos, vel, np.full(120, 0.15), floor=_box())
    move = np.linalg.norm(r.paths[:, -1] - r.paths[:, 0], axis=1)
    # A velocity-noise gate has a small false-alarm rate (a ~3σ reading in a
    # hundred); what matters is that the crowd as a whole stays put
    still = (move < 0.1).mean()
    assert still >= 0.97 and np.median(move) < 0.01, f"{still:.0%} stay; max {move.max():.1f} m"
    return f"{still:.0%} of 120 standing people stay within 10 cm over 20 s (median {np.median(move) * 100:.1f} cm)"


def test_walker_leaves_through_view_edge_and_slides_along_wall():
    floor = _box(kinds=("view", "view", "wall", "view"))   # edge 2: y = 12 (top) is a wall
    pos = np.array([[10.0, 6.0], [10.0, 10.0]])
    vel = np.array([[1.2, 0.0], [0.3, 1.2]])                # one walks right out, one into the wall
    r = CrowdForecaster().run(pos, vel, np.zeros(2), floor=floor)
    active = r.active()
    assert r.exit_step[0] > 0 and not active[0, -1], "walker 1 should leave at x = 20"
    assert (r.paths[1, :, 1] <= 12 + 1e-6).all() and r.paths[1, -1, 0] > 12, "walker 2 slides along the wall"
    return f"left the view at step {r.exit_step[0]}; the other slid to x = {r.paths[1, -1, 0]:.1f} m"


def test_head_on_walkers_avoid_each_other():
    pos = np.array([[2.0, 6.0], [18.0, 6.05]])
    vel = np.array([[1.3, 0.0], [-1.3, 0.0]])
    r = CrowdForecaster(ForecastParams(steps=20, step_s=0.4, substeps=8)).run(pos, vel, np.zeros(2))
    gap = np.linalg.norm(r.paths[0] - r.paths[1], axis=1).min()
    assert gap > 0.45, f"came within {gap:.2f} m"
    return f"closest approach {gap:.2f} m (bodies 0.5 m wide)"


def test_arrivals_follow_the_learned_rate():
    samples = [(np.array([0.2, 6.0]), np.array([1.0, 0.0]))]
    r = CrowdForecaster().run(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0), floor=_box(),
                              arrivals=(0.5, samples))
    n = len(r.paths) - r.n_tracked
    assert n == 10 and (r.enter_step[r.n_tracked:] > 0).all(), f"{n} arrivals"
    act = r.active()
    assert act[:, 0].sum() == 0 and act[:, -1].sum() > 0
    return f"{n} newcomers over 20 s at 0.5 /s"


def test_scene_memory_learns_lane_and_entries():
    floor = _box(40, 20)
    mem = SceneMemory(floor)
    rng = np.random.default_rng(0)
    people = {}
    nid = 0
    for k in range(300):                            # 60 s at 5 Hz: people walk in at x=0 along y≈10
        t = k * 0.2
        if k % 10 == 0:
            people[nid] = np.array([0.3, 10 + rng.normal(0, 0.5)])
            nid += 1
        ids = list(people)
        for i in ids:
            people[i] = people[i] + [0.26, 0]
        ids = [i for i in ids if people[i][0] < 40]
        people = {i: people[i] for i in ids}
        pos = np.array([people[i] for i in ids]).reshape(-1, 2)
        mem.observe(t, ids, pos, np.tile([1.3, 0.0], (len(ids), 1)), np.ones(len(ids)))
    d, r = mem.flow_field().at([[20.0, 10.0], [20.0, 2.0]])
    rate, samples = mem.arrivals(60.0)
    assert d[0, 0] > 0.95 and r[0] > 0.7 and r[1] < 0.2, f"lane dir {d[0]}, rel {r}"
    span = 60.0 - mem.WARMUP_S                         # the estimate is shrunk towards 0 for short histories
    expect = 0.5 * span / (span + mem.PRIOR_SPAN_S)
    assert abs(rate - expect) < 0.08 and samples[0][0][0] < 2.0, f"rate {rate:.2f} vs {expect:.2f}"
    return (f"lane reliability {r[0]:.2f} (off-lane {r[1]:.2f}); entry rate {rate:.2f}/s "
            f"(true 0.5, shrunk for 58 s of history → {expect:.2f})")


def test_skill_shrinks_a_misleading_forecast():
    rng = np.random.default_rng(0)
    sk = ForecastSkill(steps=5, step_s=1.0)
    now = np.full(6, 20.0)
    for k in range(200):                          # the agents predict big changes that never come
        t = float(k)
        truth = now + rng.normal(0, 1, 6)
        sk.observe(t, truth, tol=0.1)
        sk.add(t, now, now + np.outer(np.arange(6), rng.normal(0, 3, 6)))
    w = sk.weights()
    rep = sk.report()
    assert w[5] < 0.2, f"weight {w[5]:.2f}"
    assert rep["mae_twin"][5] < rep["mae_agents"][5], rep
    return f"weight at +5 s → {w[5]:.2f}; error {rep['mae_twin'][5]:.1f} vs raw {rep['mae_agents'][5]:.1f}"


def test_skill_keeps_a_good_forecast():
    sk = ForecastSkill(steps=5, step_s=1.0)
    rng = np.random.default_rng(1)
    for k in range(200):
        t = float(k)
        now = rng.uniform(5, 30, 6)
        change = rng.normal(0, 4, 6)
        sk.add(t, now, now + np.outer(np.arange(6) / 5, change))
        sk.pending[-1]["truth"] = now + change
    # Score them: truth at t0 + 5 s = now + change
    for f in list(sk.pending):
        sk.observe(f["t0"] + 5.0, f["truth"], tol=0.1)
    assert sk.weights()[5] > 0.9, sk.weights()
    return f"weight at +5 s stays {sk.weights()[5]:.2f}"


def _twin_with(people_fn, frames=40):
    tw = TwinPipeline(record_experience=False)
    tw.configure(H, W, 25.0)
    for f in range(frames):
        tw.process_tracks(people_fn(f), frame_idx=f)
    return tw


def test_twin_forecast_payload():
    rng = np.random.default_rng(0)
    xs, ys = rng.uniform(100, 1180, 60), rng.uniform(200, 700, 60)

    def people(f):
        return [{"id": k, "cx": float(x + 3 * f * (k % 2)), "cy": float(y) - 40, "fy": float(y),
                 "bh": 80.0, "confidence": .9} for k, (x, y) in enumerate(zip(xs, ys))]
    tw = _twin_with(people)
    svc = TwinService(tw)
    fc = svc.forecast(tw.twin_snapshot, record=True)
    json.dumps(fc)
    assert {"enter_step", "exit_step", "n_tracked", "skill"} <= set(fc)
    counts = np.array([z["counts"] for z in fc["zones"]])
    assert (counts >= 0).all() and counts.shape[1] == Config.PRED_HORIZON + 1
    walkers = [a for a in tw.dt.active_agents() if a.id % 2]
    standers = [a for a in tw.dt.active_agents() if not a.id % 2]
    assert np.mean([a.walking for a in walkers]) > 0.8 and np.mean([a.walking for a in standers]) < 0.2
    return (f"{fc['n_tracked']} people, {sum(e >= 0 for e in fc['exit_step'])} leave; "
            f"walkers flagged {np.mean([a.walking for a in walkers]):.2f}, standers "
            f"{np.mean([a.walking for a in standers]):.2f}; {fc['compute_ms']} ms")


def test_capacity_warning_when_a_crowd_moves_into_a_zone():
    """150 people walking from the bottom-left zone into the bottom-middle one (capacity 120)."""
    rng = np.random.default_rng(3)
    xs, ys = rng.uniform(10, 420, 150), rng.uniform(380, 700, 150)

    def people(f):                                   # 2.56 px per 0.08 s = 1 m/s on the flat ground
        return [{"id": k, "cx": float(x + 2.56 * f), "cy": float(y) - 40, "fy": float(y),
                 "bh": 80.0, "confidence": .9} for k, (x, y) in enumerate(zip(xs, ys))]
    tw = _twin_with(people, frames=25)
    fc = TwinService(tw).forecast(tw.twin_snapshot)
    full = {z["name"]: z["full_at_s"] for z in fc["zones"] if z["full_at_s"] is not None}
    assert "Zone_E" in full and full["Zone_E"] > 0, full
    return f"Zone E predicted to reach capacity in {full['Zone_E']} s"


def test_evaluate_forecast_on_synthetic_tracks():
    """Walkers crossing a camera view: the new twin should beat persistence and the old twin."""
    import tempfile
    from pathlib import Path
    import evaluate as E
    rng = np.random.default_rng(0)
    rows = []
    fps, n_frames = 25, 750                          # 30 s
    people = []
    for k in range(40):                              # two lanes, one each way
        start = rng.integers(0, 400)
        y = rng.uniform(15, 20) if k % 2 else rng.uniform(24, 30)   # metres from the camera
        x0, vx = (-15.0, 1.3) if k % 2 else (15.0, -1.2)
        people.append((k + 1, start, x0, y, vx))
    for f in range(1, n_frames + 1, 2):
        for pid, start, x0, y, vx in people:
            if f < start:
                continue
            x = x0 + vx * (f - start) / fps
            fx, fy = CAM.to_image([[x, y]])[0] + rng.normal(0, 0.8, 2)
            if not (0 <= fx < W and 0 <= fy < H):    # only people the camera sees
                continue
            bh = 1.7 / CAM.m_per_px([[fx, fy]])[0] * 0.55
            rows.append((f, pid, fx - bh / 5, fy - bh, bh / 2.5, bh, 1, -1, -1, -1))
    d = Path(tempfile.mkdtemp())
    np.savetxt(d / "tracks.txt", np.array(rows), delimiter=",", fmt="%.2f")
    (d / "seqinfo.ini").write_text(f"[Sequence]\nframeRate={fps}\nimWidth={W}\nimHeight={H}\n")
    cal = d / "cam.json"
    cal.write_text(json.dumps({"camera_height_m": 6.0, "pitch_deg": 25.0, "hfov_deg": 65.0}))
    src = E.TrackSource.from_image_tracks(d / "tracks.txt", str(cal))
    res = E._forecast_source(src, ForecastParams(), 2.0, 5.0, [4.8, 8.0])["models"]
    new, old, keep = res["new_twin"]["8s"], res["old_twin"]["8s"], res["persistence"]["8s"]
    assert new["fde_m"] < 0.6 * keep["fde_m"] and new["fde_m"] <= old["fde_m"] * 1.05, (new, old, keep)
    return (f"8 s FDE: new {new['fde_m']:.2f} m, old {old['fde_m']:.2f} m, "
            f"persistence {keep['fde_m']:.2f} m")


if __name__ == "__main__":
    runner.run(dict(globals()))
