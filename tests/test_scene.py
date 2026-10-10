"""
Checks for the sector-3 features: camera-motion compensation, camera pose
from point calibrations, reusing saved calibrations, learned obstacles,
doorways and the panic what-if, forecast alerts, and which stream URLs the
source switcher accepts. Needs no PyTorch:

    venv-twin\\Scripts\\python tests\\test_scene.py
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

import runner  # noqa: F401
from forecast import CrowdForecaster, Floor, ForecastParams, SceneMemory
from main import AlertEngine, Config, GroundPlane, TwinPipeline, stream_source
from stabilize import CameraMotion, apply

Config.AUTO_CALIBRATE = False
H, W = 720, 1280


def _textured(h=432, w=768, seed=0):
    """A frame with plenty of corners (stand-in for a real background)."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur((rng.random((h, w)) * 255).astype(np.uint8), (0, 0), 2)
    for _ in range(150):
        x, y = rng.integers(0, w), rng.integers(0, h)
        cv2.rectangle(img, (int(x), int(y)), (int(x) + 12, int(y) + 9), int(rng.integers(0, 255)), -1)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_camera_motion_follows_a_pan_and_ignores_a_still_camera():
    base = _textured()
    cm = CameraMotion()
    pt = np.array([[300.0, 250.0]])
    errs, first = [], None
    for f in range(30):
        M = cv2.getRotationMatrix2D((384, 216), 0.05 * f, 1.0)
        M[0, 2] += 3.0 * f                                     # a 3 px/frame pan
        T = cm.update(cv2.warpAffine(base, M, (768, 432), borderMode=cv2.BORDER_REFLECT))
        img_pt = (M[:, :2] @ pt.T).T + M[:, 2]
        if cm.moving:
            ref = apply(T, img_pt)[0]
            first = ref if first is None else first
            errs.append(np.linalg.norm(ref - first))
    assert cm.moving and max(errs) < 1.0, (cm.moving, max(errs) if errs else None)
    still = CameraMotion()
    rng = np.random.default_rng(1)
    for f in range(30):                                       # a fixed camera with sensor noise
        noisy = np.clip(base.astype(int) + rng.integers(-3, 4, base.shape), 0, 255).astype(np.uint8)
        T = still.update(noisy)
    assert not still.moving and np.allclose(T, np.eye(3))
    return f"pan tracked: a fixed point drifts {max(errs):.2f} px in the reference (90 px in the image)"


def test_camera_pose_from_a_point_calibration():
    cam = GroundPlane.from_camera(H, W, 6.0, 25.0, 65.0)
    img = np.array([[200, 700], [1100, 690], [900, 350], [350, 360], [640, 500]], float)
    gnd = cam.to_world(img)
    th = math.radians(30)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    pose = GroundPlane.from_points(H, W, img, gnd @ R.T + [12.0, -5.0]).camera()
    assert (abs(pose["height_m"] - 6) < 0.05 and abs(pose["pitch_deg"] - 25) < 0.5
            and abs(pose["hfov_deg"] - 65) < 0.5 and abs(pose["x"] - 12) < 0.1
            and abs(pose["y"] + 5) < 0.1 and abs(pose["yaw_deg"] + 30) < 0.5), pose
    assert GroundPlane.flat(H, W).camera() is None
    return f"recovered {pose}"


def test_saved_calibration_is_reused():
    name = "zz_test_reuse"
    path = Path("calibration") / f"auto_{name}.json"
    path.write_text(json.dumps({"camera_height_m": 7.5, "pitch_deg": 30, "hfov_deg": 60}))
    old = Config.AUTO_CALIBRATE
    Config.AUTO_CALIBRATE = True
    try:
        tw = TwinPipeline.__new__(TwinPipeline)
        tw.calib_name = name
        TwinPipeline.__init__(tw, record_experience=False)
        g = tw.zone_mgr.ground
        assert g.params.get("camera_height_m") == 7.5 and g.source == "pedestrians", g.describe()
        assert tw._calibrator is None                         # no need to estimate again
    finally:
        Config.AUTO_CALIBRATE = old
        path.unlink()


def _hall(door=None):
    """A 20 × 12 m hall; `door` = (x0, x1) makes the bottom a wall with that gap open."""
    if door is None:
        v, kinds = [[0, 0], [20, 0], [20, 12], [0, 12]], ["view"] * 4
    else:
        v = [[0, 0], [door[0], 0], [door[1], 0], [20, 0], [20, 12], [0, 12]]
        kinds = ["wall", "wall", "view", "wall", "wall", "wall"]
    return Floor([{"a": v[k - 1], "b": v[k], "kind": kinds[k]} for k in range(len(v))])


def test_obstacles_learned_and_walked_around():
    floor = _hall()
    mem = SceneMemory(floor)
    rng = np.random.default_rng(0)
    pillar = lambda p: (p[:, 0] > 9) & (p[:, 0] < 11) & (p[:, 1] > 5) & (p[:, 1] < 7)
    for k in range(400):
        p = rng.uniform([0.5, 0.5], [19.5, 11.5], (60, 2))
        p = p[~pillar(p)]
        mem.observe(k * 0.2, list(range(len(p))), p, np.zeros_like(p), np.zeros(len(p)))
    obs = mem.obstacles()
    assert len(obs) == 4 and pillar(obs).all(), obs
    for y0 in (6.0, 6.4, 5.3):
        r = CrowdForecaster(ForecastParams(steps=15)).run([[4.0, y0]], [[1.3, 0.0]], [0.05],
                                                          floor=floor, obstacles=obs)
        assert not pillar(r.paths[0]).any() and r.paths[0, -1, 0] > 12, (y0, r.paths[0, -1])
    return "the 2 × 2 m pillar's 4 cells found; walkers aimed at it go round"


def test_doorway_and_panic_what_if():
    floor = _hall(door=(8.5, 11.5))
    rng = np.random.default_rng(0)
    P = rng.uniform([1, 2], [19, 11], (300, 2))
    calm = CrowdForecaster().panic(P, np.zeros_like(P), np.full(300, .1), floor=floor, level=0.0, steps=100)
    wild = CrowdForecaster().panic(P, np.zeros_like(P), np.full(300, .1), floor=floor, level=0.9, steps=100)
    sc, sw = calm.summary(), wild.summary()
    through_wall = ((calm.paths[:, :, 1] < -0.3) &
                    ((calm.paths[:, :, 0] < 8.3) | (calm.paths[:, :, 0] > 11.7))).any()
    assert not through_wall
    # 300 people through a 3 m door at ≤ 1.3 people/m/s take ≥ 77 s — 90 % out after ~70 s
    assert sc["ninety_out_s"] is not None and 55 <= sc["ninety_out_s"] <= 80, sc["ninety_out_s"]
    assert sc["danger_first_s"] is not None and sw["max_density"] > sc["max_density"]
    return (f"calm: 90 % out in {sc['ninety_out_s']} s, peak {sc['max_density']} p/m²; "
            f"panic 0.9: peak {sw['max_density']} p/m², {sw['remaining'][-1]:.0%} still in after 80 s")


def test_forecast_capacity_becomes_an_alert():
    msgs = AlertEngine().generate({"Zone_A": {"risk_label": "LOW"}}, {"Zone_A": 10},
                                  {"Zone_A": (12.0, 50), "Zone_B": (0.0, 40), "Zone_C": (35.0, 40)})
    assert msgs == ["FORECAST: Zone_A predicted to reach capacity (50) in 12 s."], msgs
    tw = TwinPipeline(record_experience=False)
    tw.configure(H, W, 25.0)
    import time as _t
    tw.forecast_warnings = {"at": _t.time(), "scene_version": tw.scene_version,
                            "full": {"Zone_E": (8.0, 30)}}
    p = tw.process_tracks([{"id": 1, "cx": 640.0, "cy": 600.0, "fy": 640.0, "confidence": .9}])
    assert any(a.startswith("FORECAST: Zone_E") for a in p["alerts"]), p["alerts"]


def test_stream_urls_the_switcher_accepts():
    assert stream_source("rtsp://user:pw@10.0.0.5:554/stream") == "rtsp://user:pw@10.0.0.5:554/stream"
    assert stream_source("0") == 0 and stream_source(2) == 2
    for bad in ("videos/demo.mp4", "C:/Windows/win.ini", "file:///etc/passwd", "rtsp://a b", "99", ""):
        assert stream_source(bad) is None, bad


def test_zones_cover_equal_floor_areas():
    from main import ZoneManager
    g = GroundPlane.from_camera(596, 1280, 5.2, 19.0, 65.0)
    g.floor_top = 90.0
    zm = ZoneManager(596, 1280, g)
    areas = [z.area_m2 for z in zm.zones.values()]
    assert max(areas) - min(areas) < 0.01 * max(areas), areas
    # far zones are the top of the image, near zones the bottom; feet go to the zone they stand in
    assert zm.assign(640, 580) == "Zone_E" and zm.assign(640, 120) == "Zone_B",         (zm.assign(640, 580), zm.assign(640, 120))
    assert zm.assign(20, 590) == "Zone_D" and zm.assign(1260, 590) == "Zone_F"
    return f"6 zones of {areas[0]:.0f} m²"


def test_saved_calibration_survives_the_resize_at_start():
    """The pipeline is sized to a default frame first, then to the stream's own:
    a saved camera calibration must be kept for the real size, not dropped to flat."""
    import tempfile, os
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            Path("calibration").mkdir()
            Path("calibration/auto_clip.json").write_text(json.dumps(
                {"camera_height_m": 5.2, "pitch_deg": 19.0, "hfov_deg": 65.0, "image_size": [1280, 596]}))
            Config.AUTO_CALIBRATE = True
            tw = TwinPipeline(record_experience=False)
            tw.calib_name, tw._auto_tried, tw._auto_ground = "clip", False, None
            tw.configure(720, 1280, 25.0)
            tw.configure(596, 1280, 30.0)
            g = tw.zone_mgr.ground
            assert g.params.get("camera_height_m") == 5.2 and (g.h, g.w) == (596, 1280), g.params
        finally:
            Config.AUTO_CALIBRATE = False
            os.chdir(old_cwd)


def test_portrait_video_assumes_the_narrow_field_of_view():
    from main import assumed_hfov
    assert assumed_hfov(720, 1280) == Config.CAMERA_HFOV_DEG
    assert 38 < assumed_hfov(1280, 720) < 40          # 65° across the long side


if __name__ == "__main__":
    runner.run(dict(globals()))
