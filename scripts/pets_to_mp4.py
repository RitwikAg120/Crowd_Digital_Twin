"""
Turn a multi-camera image-sequence dataset (e.g. PETS 2009, View_001 … View_008)
into one video per camera, for the multi-camera demo:

    python scripts/pets_to_mp4.py <folder with View_* sub-folders> [out folder] [fps]
    python scripts/pets_to_mp4.py pets/Crowd_PETS09/S2/L1/Time_12-34 videos/pets 7

    python main.py --source videos/pets/pets_View_001.mp4 --source videos/pets/pets_View_005.mp4

PETS 2009 was recorded at ~7 fps. All views of a sequence start at the same
instant, so the videos play in step.
"""
import glob
import os
import sys

import cv2

src = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else "videos/pets"
fps = float(sys.argv[3]) if len(sys.argv) > 3 else 7.0
os.makedirs(out_dir, exist_ok=True)

views = sorted(d for d in glob.glob(os.path.join(src, "View_*")) if os.path.isdir(d))
if not views:
    raise SystemExit(f"No View_* folders in {src}")
for view in views:
    frames = sorted(glob.glob(os.path.join(view, "*.jpg")))
    if not frames:
        continue
    h, w = cv2.imread(frames[0]).shape[:2]
    out = os.path.join(out_dir, f"pets_{os.path.basename(view)}.mp4")
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.imread(f))
    vw.release()
    print(f"{out}: {len(frames)} frames, {w}x{h}")
