"""
Checks for multi-camera support: placing cameras on one site map, counting
people in overlapping views once, and switching the camera on screen.
Needs no PyTorch:

    venv-twin\\Scripts\\python tests\\test_multicam.py
"""
import numpy as np

import runner  # noqa: F401
import main
from main import Config, TwinPipeline, TwinService
from multicam import SiteFusion

Config.AUTO_CALIBRATE = False


def test_cameras_are_placed_on_the_site_map():
    s = SiteFusion({"a": {"x": 0, "y": 0, "yaw_deg": 0},
                    "b": {"x": 10, "y": 20, "yaw_deg": 180}})
    # a looks along +Y: its (1, 5) is site (1, 5); b looks back along −Y from (10, 20)
    assert np.allclose(s.to_site("a", [[1, 5]]), [[1, 5]])
    assert np.allclose(s.to_site("b", [[1, 5]]), [[9, 15]])
    s90 = SiteFusion({"c": {"x": 0, "y": 0, "yaw_deg": 90}})       # looks along −X
    assert np.allclose(s90.to_site("c", [[0, 4]]), [[-4, 0]])


def test_people_in_overlapping_views_are_counted_once():
    s = SiteFusion({"a": {"x": 0, "y": 0, "yaw_deg": 0},
                    "b": {"x": 0, "y": 20, "yaw_deg": 180}}, merge_radius_m=0.6)
    rng = np.random.default_rng(0)
    shared = rng.uniform([-3, 8], [3, 12], (10, 2))                 # the overlap, site metres
    only_a = rng.uniform([-3, 2], [3, 6], (5, 2))
    only_b = rng.uniform([-3, 14], [3, 18], (7, 2))
    # each camera sees its people in its own frame (b: rotated 180° about (0, 20)) with 0.2 m noise
    in_b = lambda p: np.column_stack([-p[:, 0], 20 - p[:, 1]])
    seen_a = np.vstack([only_a, shared + rng.normal(0, 0.1, shared.shape)])
    seen_b = np.vstack([in_b(only_b), in_b(shared + rng.normal(0, 0.1, shared.shape))])
    r = s.fuse({"a": seen_a, "b": seen_b})
    assert r["per_camera"] == {"a": 15, "b": 17}
    assert r["unique_people"] == 22 and r["seen_twice"] == 10, r
    # without a site file nothing is merged
    assert SiteFusion().fuse({"a": seen_a, "b": seen_b})["unique_people"] == 32
    return f"{r['per_camera']} → {r['unique_people']} unique"


def _pipeline(n, x0, cid):
    p = TwinPipeline(record_experience=False)
    p.configure(720, 1280, 25.0)
    p.camera_id = cid
    for f in range(5):
        p.process_tracks([{"id": k, "cx": float(x0 + 40 * k), "cy": 560.0, "fy": 600.0,
                           "bh": 120.0, "confidence": .9} for k in range(n)], frame_idx=f)
    return p


def test_switching_the_camera_on_screen():
    from fastapi.testclient import TestClient
    a, b = _pipeline(6, 100, "cam1"), _pipeline(9, 200, "cam2")
    main.cameras.clear()
    main.cameras.update({"cam1": a, "cam2": b})
    main.twin_services.update({"cam1": TwinService(a), "cam2": TwinService(b)})
    a.on_air, b.on_air = True, False
    main.pipeline, main.twin_service = a, main.twin_services["cam1"]
    try:
        c = TestClient(main.app)
        cams = c.get("/api/cameras").json()["cameras"]
        assert [k["id"] for k in cams] == ["cam1", "cam2"] and cams[0]["on_air"], cams
        assert c.post("/api/camera", json={"id": "cam2"}).json() == {"on_air": "cam2"}
        assert main.pipeline is b and b.on_air and not a.on_air
        assert c.get("/api/snapshot").json()["n_agents"] == 9
        assert c.post("/api/camera", json={"id": "nope"}).status_code == 404
        site = c.get("/api/site").json()
        assert site["unique_people"] == 15 and site["per_camera"] == {"cam1": 6, "cam2": 9}, site
    finally:
        main.cameras.clear()
        main.twin_services.clear()


if __name__ == "__main__":
    runner.run(dict(globals()))
