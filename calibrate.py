"""
Ground calibration for the Crowd Digital Twin (perspective).

Click at least 4 points on the ground in a frame of the video, type where each
point is on the ground in metres, and the result is saved as JSON for
`main.py --calibration`:

  python calibrate.py --source videos/demo.mp4 --out calibration/demo.json

Good points are floor tiles, lane or parking markings, the corners of a mat or
stage — anything whose real distances you know or can pace out. Spread them
over the area people walk in. In the window: left-click adds a point,
Backspace removes the last one, Enter finishes, Esc cancels. (The click window
needs the full `venv`; `venv-twin` has no GUI.)

If you know how the camera is mounted, write the file directly instead:

  python calibrate.py --camera-height 6.5 --pitch 30 --hfov 70 --out calibration/cam1.json

Without any calibration, main.py estimates the camera from people's heights.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from main import Config, GroundPlane, ZoneManager, fit_within


def read_frame(source: str, frame_no: int) -> np.ndarray:
    """One frame at the pipeline's working resolution."""
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {source}")
    if frame_no:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Cannot read frame {frame_no} of {source}")
    h, w = fit_within(*frame.shape[:2], Config.FRAME_HEIGHT, Config.FRAME_WIDTH)
    return cv2.resize(frame, (w, h))


def pick_points(frame: np.ndarray) -> list:
    """Let the user click ground points; returns [(x, y), ...] in pixels."""
    pts = []
    win = "Click ground points - Enter: done, Backspace: undo, Esc: cancel"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = frame.copy()
        for i, (x, y) in enumerate(pts, 1):
            cv2.circle(view, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(view, str(i), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if key in (10, 13):
            if len(pts) >= 4:
                break
            print("Click at least 4 points first.")
        elif key == 8 and pts:
            pts.pop()
        elif key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled.")
    cv2.destroyAllWindows()
    return pts


def ask_world(pts: list) -> list:
    """Ask for each point's ground position (X Y in metres)."""
    print("Where is each point on the ground? Type X Y in metres, in any frame of "
          "reference — e.g. one corner at 0 0, X along one edge, Y along the other.")
    world = []
    for i, (x, y) in enumerate(pts, 1):
        while True:
            try:
                X, Y = (float(v) for v in
                        input(f"  point {i} (pixel {x}, {y}): ").replace(",", " ").split())
                world.append((X, Y))
                break
            except ValueError:
                print("    type two numbers, e.g. 3.5 12")
    return world


def save_points(image_pts, world_pts, size, out) -> GroundPlane:
    """Save a point calibration (image_size = (w, h)) and return its ground plane."""
    w, h = size
    ground = GroundPlane.from_points(h, w, image_pts, world_pts)
    data = {
        "image_size":   [w, h],
        "image_points": [[float(v) for v in p] for p in image_pts],
        "world_points": [[float(v) for v in p] for p in world_pts],
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return ground


def summary(ground: GroundPlane) -> str:
    """Ground area in view and per zone, to sanity-check a calibration."""
    zones = ZoneManager(ground.h, ground.w, ground).zones
    areas = ", ".join(f"{n[-1]} {z.area_m2:.0f}" for n, z in zones.items())
    return (f"visible ground ≈ {sum(z.area_m2 for z in zones.values()):.0f} m² "
            f"(zones {areas} m²)")


def main():
    p = argparse.ArgumentParser(description="Ground calibration for the Crowd Digital Twin")
    p.add_argument("--out", required=True, help="calibration JSON to write")
    p.add_argument("--source", help="video, webcam index or RTSP URL to click points on")
    p.add_argument("--frame", type=int, default=0, help="frame number to show")
    p.add_argument("--camera-height", type=float, help="camera height above the ground (m)")
    p.add_argument("--pitch", type=float, help="camera tilt below horizontal (degrees)")
    p.add_argument("--hfov", type=float, default=Config.CAMERA_HFOV_DEG,
                   help="horizontal field of view (degrees)")
    args = p.parse_args()

    if args.camera_height is not None:
        if args.pitch is None:
            p.error("--pitch is needed with --camera-height")
        data = {"camera_height_m": args.camera_height, "pitch_deg": args.pitch,
                "hfov_deg": args.hfov}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(data, indent=2), encoding="utf-8")
        ground = GroundPlane.from_camera(Config.FRAME_HEIGHT, Config.FRAME_WIDTH,
                                         args.camera_height, args.pitch, args.hfov)
        print(f"Saved {args.out}: {summary(ground)} for a 16:9 view — far zones shrink "
              f"once main.py learns from people's feet where the floor ends")
        return

    if not args.source:
        p.error("give --source to click points, or --camera-height and --pitch")
    frame = read_frame(args.source, args.frame)
    pts   = pick_points(frame)
    world = ask_world(pts)
    ground = save_points(pts, world, frame.shape[1::-1], args.out)
    err = np.linalg.norm(ground.to_world(pts) - np.asarray(world, float), axis=1)
    fit = (f"points fit within {err.max():.2f} m" if len(pts) > 4
           else "4 points fit exactly — add more to check them")
    print(f"Saved {args.out}: {summary(ground)}; {fit}")
    print(f"Use it with:  python main.py --source {args.source} --calibration {args.out}")


if __name__ == "__main__":
    main()
