"""
Checks for the server's REST endpoints: gate counts (IoT), health, sources
and the 3D twin. Uses FastAPI's TestClient (needs httpx — in the full
environment):

    venv\\Scripts\\python tests\\test_api.py
"""
import numpy as np

import runner  # noqa: F401
import main
from iot import parse_counts
from main import Config, TwinPipeline, TwinService

Config.AUTO_CALIBRATE = False       # the checks must not write calibration files into the project


def _client():
    from fastapi.testclient import TestClient
    p = TwinPipeline(record_experience=False)
    p.configure(720, 1280, 25.0)
    rng = np.random.default_rng(0)
    xs, ys = rng.uniform(100, 1100, 30), rng.uniform(300, 700, 30)
    for f in range(20):
        p.process_tracks([{"id": k, "cx": float(x + 2 * f), "cy": float(y) - 40, "fy": float(y),
                           "bh": 80.0, "confidence": .9} for k, (x, y) in enumerate(zip(xs, ys))],
                         frame_idx=f)
    main.pipeline, main.twin_service = p, TwinService(p)
    return TestClient(main.app), p            # no `with`: the startup hook (which starts video) doesn't run


def test_parse_counts():
    assert parse_counts({"entry": 3, "exit": 1}) == (3, 1)
    assert parse_counts("4,2") == (4, 2)
    assert parse_counts(b'{"entry": 2}') == (2, 0)
    for bad in ({"entry": -1}, "x,y", {"entry": 10**6}):
        try:
            parse_counts(bad)
        except (ValueError, TypeError):
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_iot_endpoint_switches_fusion_to_real_gates():
    c, p = _client()
    assert p.fusion.iot.mode == "simulated"
    r = c.post("/api/iot", json={"entry": 5, "exit": 2})
    assert r.status_code == 200 and r.json()["mode"] == "live", r.text
    assert p.fusion.iot.net_count == 3
    assert c.post("/api/iot", json={"entry": "lots"}).status_code == 422
    return f"gates live, net {p.fusion.iot.net_count}"


def test_iot_token():
    c, p = _client()
    Config.IOT_TOKEN = "s3cret"
    try:
        assert c.post("/api/iot", json={"entry": 1, "exit": 0}).status_code == 401
        assert c.post("/api/iot", json={"entry": 1, "exit": 0},
                      headers={"X-IoT-Token": "s3cret"}).status_code == 200
    finally:
        Config.IOT_TOKEN = None


def test_health_and_sources_in_twin_only_mode():
    c, p = _client()
    h = c.get("/api/health").json()
    assert h["status"] == "ok" and h["people"] == 30 and h["iot"] == "simulated", h
    s = c.get("/api/sources").json()
    assert s["switchable"] is False
    assert c.post("/api/source", json={"name": "demo.mp4"}).status_code == 409
    return f"health {h['status']}, {h['people']} people"


def test_twin_endpoint_has_calibrated_forecast():
    c, p = _client()
    r = c.get("/api/twin").json()
    fc = r["forecast"]
    assert r["scene"]["type"] == "scene" and fc["type"] == "forecast"
    assert len(fc["exit_step"]) == len(fc["paths"]) and "skill" in fc
    assert all(len(z["counts"]) == Config.PRED_HORIZON + 1 for z in fc["zones"])


def test_live_stream_reconnects():
    """A live source that stops delivering frames is reopened by itself."""
    import time
    from pathlib import Path
    from main import VideoInputHandler
    if not Path("videos/demo.mp4").exists():
        return "skipped (no videos/demo.mp4)"

    class Dead:                                   # a camera / network that went away
        def read(self):
            return False, None

        def release(self):
            pass

    old = Config.STREAM_RETRY_READS
    Config.STREAM_RETRY_READS = 20
    v = VideoInputHandler("videos/demo.mp4")
    v._is_file = False                            # behave like an RTSP stream
    try:
        v.start()
        time.sleep(0.5)
        seq0 = v.read()[1]
        v.cap = Dead()
        deadline = time.time() + 15
        while v.reconnects == 0 and time.time() < deadline:
            time.sleep(0.1)
        time.sleep(0.5)
        assert v.reconnects == 1 and v.read()[1] > seq0, (v.reconnects, seq0)
    finally:
        v.stop()
        Config.STREAM_RETRY_READS = old
    return "stream reopened after it stopped delivering frames"


if __name__ == "__main__":
    runner.run(dict(globals()))
