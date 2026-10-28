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


def _two_cameras():
    a, b = _pipeline(6, 100, "cam1"), _pipeline(9, 200, "cam2")
    main.cameras.clear()
    main.cameras.update({"cam1": a, "cam2": b})
    main.twin_services.update({"cam1": TwinService(a), "cam2": TwinService(b)})
    a.on_air, b.on_air = True, False
    main.pipeline, main.twin_service = a, main.twin_services["cam1"]
    return a, b


def test_grid_has_a_tile_for_every_camera_not_only_the_one_on_screen():
    from fastapi.testclient import TestClient
    a, b = _two_cameras()
    try:
        with TestClient(main.app).websocket_connect("/ws/grid") as ws:
            tiles = {t["id"]: t for t in (ws.receive_json(), ws.receive_json())}
        assert set(tiles) == {"cam1", "cam2"}, tiles.keys()
        assert (tiles["cam1"]["people"], tiles["cam2"]["people"]) == (6, 9)
        assert tiles["cam1"]["on_air"] and not tiles["cam2"]["on_air"]
        assert {"risk", "fps", "zones", "alerts", "thumb_b64"} <= set(tiles["cam2"])
        assert not main.grid_manager.active            # the socket is gone: no more thumbnails
    finally:
        main.cameras.clear()
        main.twin_services.clear()


def test_tile_image_is_made_only_while_the_grid_is_watched():
    a = _pipeline(3, 100, "cam1")
    assert a._thumb_b64 is None
    main.grid_manager.active.append(object())          # someone opens the grid
    try:
        a.process_tracks([{"id": 1, "cx": 300.0, "cy": 560.0, "fy": 600.0, "bh": 120.0,
                           "confidence": .9}], frame=np.zeros((720, 1280, 3), np.uint8), frame_idx=9)
    finally:
        main.grid_manager.active.clear()
    assert a._thumb_b64 and a.grid_tile()["thumb_b64"] == a._thumb_b64
    import base64
    import cv2
    img = cv2.imdecode(np.frombuffer(base64.b64decode(a._thumb_b64), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == Config.GRID_THUMB_W, img.shape


def test_removing_and_adding_cameras():
    from fastapi.testclient import TestClient
    a, b = _two_cameras()
    try:
        c = TestClient(main.app)
        # a live run (CDTPipeline) can take more cameras; this stand-in pipeline can't
        assert c.post("/api/cameras", json={"name": "x.mp4"}).status_code == 409
        assert c.get("/api/cameras").json()["addable"] is False
        assert c.delete("/api/cameras/nope").status_code == 404
        r = c.delete("/api/cameras/cam1").json()          # cam1 is on screen: cam2 takes over
        assert r == {"removed": "cam1", "on_air": "cam2"}, r
        assert main.pipeline is b and b.on_air and "cam1" not in main.twin_services
        assert [k["id"] for k in c.get("/api/cameras").json()["cameras"]] == ["cam2"]
        assert c.delete("/api/cameras/cam2").status_code == 409     # never the last one
    finally:
        main.cameras.clear()
        main.twin_services.clear()


if __name__ == "__main__":
    runner.run(dict(globals()))
