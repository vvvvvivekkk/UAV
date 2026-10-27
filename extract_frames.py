"""Save one frame every EVERY_SECONDS from the video into my-dataset/images/."""

from pathlib import Path

import cv2

VIDEO = "videos/street.mp4"
OUT = Path("my-dataset/images")
EVERY_SECONDS = 1.0  # lower this (e.g. 0.6) if you want more frames

OUT.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
step = max(1, round(fps * EVERY_SECONDS))  # take every Nth frame

saved = index = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if index % step == 0:
        saved += 1
        cv2.imwrite(str(OUT / f"frame_{saved:04d}.jpg"), frame)
    index += 1
cap.release()

print(f"{fps:g} fps -> keeping every {step}th frame")
print(f"read {index} frames, saved {saved} to {OUT}")
