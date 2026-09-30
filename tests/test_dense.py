"""
Checks for dense-crowd support: feet from heads, calibration from heads,
heads without bodies becoming people, and the P2PNet point model (its
checkpoint format, inference and a training step). The model checks need
PyTorch (the full environment):

    venv\\Scripts\\python tests\\test_dense.py
"""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import runner  # noqa: F401
from main import CDTPipeline, Config, GroundPlane, PedestrianCalibrator

H, W = 432, 768
CAM = GroundPlane.from_camera(H, W, 8.0, 28.0, 65.0)
LIFT = Config.PERSON_HEIGHT_M - Config.HEAD_SIZE_M


def _head_box(cam, x, y):
    """A person standing at ground (x, y): their head box and feet pixel."""
    chin_plane = GroundPlane.from_camera(cam.h, cam.w, cam.params["camera_height_m"] - LIFT,
                                         cam.params["pitch_deg"], cam.params["hfov_deg"])
    top_plane = GroundPlane.from_camera(cam.h, cam.w, cam.params["camera_height_m"] - Config.PERSON_HEIGHT_M,
                                        cam.params["pitch_deg"], cam.params["hfov_deg"])
    cu, cv = chin_plane.to_image([[x, y]])[0]
    tu, tv = top_plane.to_image([[x, y]])[0]
    hh = cv - tv
    return [cu - hh * 0.4, tv, cu + hh * 0.4, cv], cam.to_image([[x, y]])[0]


def test_feet_from_heads_with_a_camera_model():
    rng = np.random.default_rng(0)
    boxes, feet = zip(*[_head_box(CAM, x, y) for x, y in
                        zip(rng.uniform(-6, 6, 50), rng.uniform(12, 40, 50))])
    got, body_h = CAM.feet_from_heads(np.array(boxes))
    err = np.linalg.norm(got - np.array(feet), axis=1)
    assert err.max() < 0.5 and (body_h > 0).all(), f"max error {err.max():.2f} px"
    return f"feet placed within {err.max():.2f} px of the truth for 50 heads"


def test_feet_from_heads_without_a_camera_model():
    flat = GroundPlane.flat(H, W)
    feet, bh = flat.feet_from_heads([[100, 100, 110, 112]])
    assert abs(feet[0, 1] - (112 + (Config.BODY_PER_HEAD - 1) * 12)) < 1e-9 and abs(bh[0] - 7 * 12) < 1e-9


def test_calibration_from_heads_only():
    rng = np.random.default_rng(1)
    cal = PedestrianCalibrator(H, W)
    for _ in range(30):
        heads = []
        for x, y in zip(rng.uniform(-8, 8, 40), rng.uniform(10, 45, 40)):
            b, _ = _head_box(CAM, x, y)
            h = b[3] - b[1]
            b = np.array(b) + np.r_[rng.normal(0, 0.03 * h, 4)]
            if b[1] > 3 and b[3] < H - 3:
                heads.append({"box": b.tolist(), "confidence": 0.8})
        cal.add([], heads)
    assert cal.ready
    ground, why = cal.fit()
    ch, pitch = ground.params["camera_height_m"], ground.params["pitch_deg"]
    assert abs(ch - 8.0) < 1.2 and abs(pitch - 28.0) < 4, why
    return why


def test_heads_without_bodies_become_people():
    fake = SimpleNamespace(_inside_any=CDTPipeline._inside_any)
    bodies = np.array([[100, 100, 140, 220]], np.float32)
    heads = [{"box": [112, 102, 126, 118], "confidence": 0.6},     # this body's own head
             {"box": [300, 50, 310, 62], "confidence": 0.3}]       # someone the body detector missed
    boxes, conf, src = CDTPipeline._unmatched_heads(fake, bodies, heads)
    assert len(boxes) == 1 and boxes[0][0] == 300 and src == ["head"]
    pts = (np.array([[120.0, 110.0], [400.0, 60.0], [410.0, 62.0]]), np.array([0.9, 0.8, 0.7]))
    boxes, conf, src = CDTPipeline._unmatched_heads(fake, bodies, heads, pts)
    assert len(boxes) == 2 and src == ["point", "point"]


def test_head_points_sized_by_perspective():
    from dense import HeadScale
    rng = np.random.default_rng(2)
    hs = HeadScale()
    assert hs.predict([100]) is None
    heads = []
    for y in rng.uniform(40, 420, 300):                   # heads grow 3 px per 100 rows
        s = 0.03 * y + 4 + rng.normal(0, 0.4)
        heads.append({"box": [200, y - s / 2, 200 + s, y + s / 2], "confidence": 0.7})
    heads += [{"box": [0, y, 70, y + 70], "confidence": 0.9} for y in (80, 160, 240)]   # false hits
    hs.add(heads)
    got = hs.predict([50, 400])
    assert np.allclose(got, [5.5, 16.0], atol=0.6), got
    # Sparse points (the point model missed their neighbours) still get head-sized boxes
    fake = SimpleNamespace(_inside_any=CDTPipeline._inside_any)
    pts = (np.array([[100.0, 50.0], [600.0, 400.0]]), np.array([0.9, 0.8]))
    boxes, _, _ = CDTPipeline._unmatched_heads(fake, np.zeros((0, 4), np.float32), [], pts, hs)
    assert np.allclose(boxes[:, 2] - boxes[:, 0], got, atol=1e-3)
    return f"head size {got[0]:.1f} px at row 50, {got[1]:.1f} px at row 400"


def test_heads_are_tracked_by_their_heads():
    boxes = np.array([[100, 100, 110, 112]], np.float32)
    big = CDTPipeline._scale_boxes(boxes, 2.0)   # (any factor)
    assert np.allclose(big, [[95, 94, 115, 118]])
    assert np.allclose(CDTPipeline._scale_boxes(big, 0.5), boxes)
    fake = SimpleNamespace(zone_mgr=SimpleNamespace(ground=GroundPlane.flat(H, W)))
    person = CDTPipeline._person_boxes(fake, boxes)
    assert person[0, 1] == 100 and abs(person[0, 3] - (112 + (Config.BODY_PER_HEAD - 1) * 12)) < 1e-3


def _torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def test_p2pnet_checkpoint_format_and_inference():
    if not _torch():
        return "skipped (no PyTorch in this environment)"
    import torch
    from dense import P2PNet, PointCounter
    m = P2PNet()
    keys = set(m.state_dict())
    # Names in the official Tencent checkpoint
    for k in ("backbone.body1.0.weight", "backbone.body4.8.running_var", "fpn.P5_1.weight",
              "fpn.P3_2.bias", "regression.conv3.weight", "classification.output.bias"):
        assert k in keys, k
    x = torch.zeros(1, 3, 256, 384)
    logits, pts = m.eval()(x)
    assert logits.shape == (1, 32 * 48 * 4, 2) and pts.shape == (1, 32 * 48 * 4, 2)
    path = Path(tempfile.mkdtemp()) / "p2p.pth"
    torch.save({"model": {"module." + k: v for k, v in m.state_dict().items()}}, path)
    pc = PointCounter(path, threshold=0.0)
    img = (np.random.default_rng(0).random((200, 300, 3)) * 255).astype(np.uint8)
    p, s = pc(img)
    assert len(p) and (p[:, 0] >= 0).all() and (p[:, 0] < 300).all() and (p[:, 1] < 200).all()
    return f"official key layout; {len(p)} candidate points inside a 300×200 frame"


def test_train_dense_one_step():
    if not _torch():
        return "skipped (no PyTorch in this environment)"
    import cv2
    import subprocess
    import sys
    d = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    for split, n in (("train", 2), ("test", 1)):
        (d / split).mkdir()
        for i in range(n):
            img = np.full((160, 224, 3), 160, np.uint8)
            pts = rng.uniform([8, 8], [216, 152], (30, 2))
            for x, y in pts:
                cv2.circle(img, (int(x), int(y)), 3, (30, 30, 30), -1)
            cv2.imwrite(str(d / split / f"{i}.jpg"), img)
            np.savetxt(d / split / f"{i}.txt", pts)
    out = d / "m.pth"
    r = subprocess.run([sys.executable, "train_dense.py", "--data", f"points:{d}", "--init", "none",
                        "--epochs", "1", "--batch", "2", "--patches", "1", "--crop", "128",
                        "--workers", "0", "--device", "cpu", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and out.exists(), r.stderr[-500:]
    return r.stdout.strip().splitlines()[-1]


def test_crowdhuman_heads_and_ignore_regions():
    if not _torch():
        return "skipped (no PyTorch in this environment)"
    import json
    import cv2
    from train_dense import list_samples, load_image
    root = Path(tempfile.mkdtemp())
    (root / "Images").mkdir()
    cv2.imwrite(str(root / "Images" / "a1,b.jpg"), np.full((100, 200, 3), 200, np.uint8))
    rec = {"ID": "a1,b", "gtboxes": [
        {"tag": "person", "hbox": [10, 10, 10, 10], "head_attr": {"ignore": 0}},
        {"tag": "person", "hbox": [50, 10, 10, 10], "head_attr": {"ignore": 1}},
        {"tag": "mask", "fbox": [150, 50, 40, 40]}]}
    for split in ("train", "val"):
        (root / f"annotation_{split}.odgt").write_text(json.dumps(rec))
    (path, pts, ignore), = list_samples("crowdhuman", str(root), "train")
    assert np.allclose(pts, [[15, 15]]) and len(ignore) == 2
    img, _ = load_image(path, pts, 2048, ignore)
    assert (img[60, 160] != 200).all() and (img[80, 100] == 200).all()   # the mask is blanked
    assert len(list_samples("crowdhuman", str(root), "test")) == 1


if __name__ == "__main__":
    runner.run(dict(globals()))
