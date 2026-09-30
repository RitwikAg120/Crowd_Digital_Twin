"""
Export the detectors — and the dense-crowd point model — for edge devices.
Run it ON the target (Jetson or GB10): a TensorRT engine is built for the GPU
and TensorRT version it runs on.

    python export.py                       # TensorRT FP16 engines (default)
    python export.py --format onnx         # portable ONNX
    python export.py --int8 --data <yaml>  # TensorRT INT8 (needs calibration images)

Then run with the exported files, e.g.
    python main.py --source rtsp://... --body weights/yolo26strained.engine \
                   --head weights/yoloheadv26s.engine
(Ultralytics loads .engine / .onnx like .pt.) The head engine is built with
dynamic input sizes up to the dense-mode resolution (HBOX_IMGSZ_DENSE).
Then measure: python evaluate.py bench --source videos/demo.mp4
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")

from main import Config


def export_yolo(path: str, fmt: str, imgsz: int, half: bool, int8: bool, data, dynamic: bool):
    from ultralytics import YOLO
    out = YOLO(path).export(format=fmt, imgsz=imgsz, half=half and not int8, int8=int8,
                            data=data, dynamic=dynamic, device=0 if fmt == "engine" else "cpu",
                            batch=1, simplify=True)
    print(f"  {path} → {out}")
    return out


def export_points(path: str, out: Path, h: int = 768, w: int = 1280):
    """P2PNet → ONNX with dynamic height/width (multiples of 128)."""
    import torch
    from dense import load_p2pnet
    model = load_p2pnet(path, "cpu")
    x = torch.zeros(1, 3, h, w)
    torch.onnx.export(model, x, str(out), input_names=["image"], output_names=["logits", "points"],
                      dynamic_axes={"image": {2: "h", 3: "w"}, "logits": {1: "k"}, "points": {1: "k"}},
                      opset_version=17)
    print(f"  {path} → {out}  (build a TensorRT engine with: trtexec --onnx={out} --fp16 "
          f"--minShapes=image:1x3x128x128 --optShapes=image:1x3x{h}x{w} "
          f"--maxShapes=image:1x3x1536x2560 --saveEngine={out.with_suffix('.engine')})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", default="engine", choices=["engine", "onnx"])
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--data", help="dataset YAML with calibration images (INT8)")
    ap.add_argument("--no-half", action="store_true")
    args = ap.parse_args()
    half = not args.no_half
    print("Body detector")
    export_yolo(Config.YOLO_MODEL, args.format, Config.YOLO_IMGSZ, half, args.int8, args.data, False)
    print("Head detector (dynamic sizes up to the dense-mode input)")
    export_yolo(Config.HBOX_MODEL, args.format, Config.HBOX_IMGSZ_DENSE, half, args.int8, args.data, True)
    if Config.DENSE_MODEL and Path(Config.DENSE_MODEL).exists():
        print("Dense-crowd point model")
        export_points(Config.DENSE_MODEL, Path(Config.DENSE_MODEL).with_suffix(".onnx"))


if __name__ == "__main__":
    main()
