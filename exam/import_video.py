r"""
Turn a phone video into training photos.

Usage (from the UAV folder):
  .venv2\Scripts\python.exe exam\import_video.py <video file> <name>
Example:
  .venv2\Scripts\python.exe exam\import_video.py C:\Users\11020\Downloads\praney.mp4 praney

Saves 1 photo every EVERY_SECONDS into exam\dataset_raw\<name>\
"""
import sys
from pathlib import Path

import cv2

EVERY_SECONDS = 0.3      # one photo every 0.3 s of video
MAX_WIDTH = 960          # shrink big phone frames to about Tello size

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(1)

video, name = sys.argv[1], sys.argv[2]
out = Path(__file__).parent / "dataset_raw" / name
out.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    sys.exit(f"Could not open video: {video}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
step = max(1, int(fps * EVERY_SECONDS))

i = saved = 0
stem = Path(video).stem
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if i % step == 0:
        h, w = frame.shape[:2]
        if w > MAX_WIDTH:
            frame = cv2.resize(frame, (MAX_WIDTH, int(h * MAX_WIDTH / w)))
        cv2.imwrite(str(out / f"{name}_{stem}_{i:05d}.jpg"), frame)
        saved += 1
    i += 1
cap.release()
print(f"Saved {saved} photos to {out}")
