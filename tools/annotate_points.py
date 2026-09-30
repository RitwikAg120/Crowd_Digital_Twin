"""
Click head points on video frames — training data for the dense-crowd model
in the "points" format train_dense.py reads (<name>.jpg + <name>.txt, one
"x y" per head).

    venv\\Scripts\\python tools/annotate_points.py --video videos/kumbhvideo4.mp4 \\
        --every 75 --out dataset/kumbh_points --split train --prefill heads

  left click   add a head            right click   remove the nearest head
  z            undo                  + / -         zoom in / out (for tiny heads)
  n or space   save, next frame      s             skip this frame
  q / Esc      save and quit

--prefill heads starts each frame from the head detector at dense-mode
resolution (or --prefill <p2pnet.pth> from a point model), so you only fix
misses and false heads. Keep test frames from different clips than train.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def prefill_fn(kind):
    if not kind:
        return lambda img: np.zeros((0, 2))
    os.environ.setdefault("YOLO_OFFLINE", "1")
    from main import Config
    if kind == "heads":
        from ultralytics import YOLO
        model = YOLO(Config.HBOX_MODEL)

        def heads(img):
            r = model(img, conf=Config.HBOX_CONF_DENSE, imgsz=Config.HBOX_IMGSZ_DENSE,
                      max_det=5000, verbose=False)[0]
            b = r.boxes.xyxy.cpu().numpy()
            return np.column_stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2])
        return heads
    from dense import PointCounter
    pc = PointCounter(kind)
    return lambda img: pc(img)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--every", type=int, default=75, help="annotate every Nth frame")
    ap.add_argument("--out", default="dataset/kumbh_points")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--prefill", help="'heads' or a point-model checkpoint")
    args = ap.parse_args()

    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    predict = prefill_fn(args.prefill)
    cap = cv2.VideoCapture(args.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    state = {"pts": [], "zoom": 2.0 if cap.get(cv2.CAP_PROP_FRAME_WIDTH) < 1000 else 1.0}
    win = "heads — left add, right remove, n next, q quit"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def on_mouse(ev, x, y, flags, _):
        z = state["zoom"]
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["pts"].append((x / z, y / z))
        elif ev == cv2.EVENT_RBUTTONDOWN and state["pts"]:
            p = np.array(state["pts"])
            state["pts"].pop(int(np.argmin(np.hypot(p[:, 0] - x / z, p[:, 1] - y / z))))
    cv2.setMouseCallback(win, on_mouse)

    stem = Path(args.video).stem
    for f in range(0, n, args.every):
        name = f"{stem}_{f:06d}"
        if (out / f"{name}.txt").exists():
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            break
        state["pts"] = [tuple(p) for p in predict(img)]
        while True:
            z = state["zoom"]
            view = cv2.resize(img, None, fx=z, fy=z, interpolation=cv2.INTER_LINEAR)
            for x, y in state["pts"]:
                cv2.circle(view, (int(x * z), int(y * z)), max(2, int(2 * z)), (0, 0, 255), -1)
            cv2.putText(view, f"{name}: {len(state['pts'])} heads", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(win, view)
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("n"), ord(" "), ord("q"), 27):
                cv2.imwrite(str(out / f"{name}.jpg"), img)
                np.savetxt(out / f"{name}.txt", np.array(state["pts"]).reshape(-1, 2), fmt="%.1f")
                print(f"{name}: {len(state['pts'])} heads saved")
                if k in (ord("q"), 27):
                    return
                break
            if k == ord("s"):
                break
            if k == ord("z") and state["pts"]:
                state["pts"].pop()
            if k in (ord("+"), ord("=")):
                state["zoom"] = min(4.0, z + 0.5)
            if k == ord("-"):
                state["zoom"] = max(1.0, z - 0.5)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
