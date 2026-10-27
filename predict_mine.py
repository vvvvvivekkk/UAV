"""Run YOUR trained model (best.pt) on the street video, and print the same
stats video_demo.py printed for yolo26n.pt so the two can be compared."""

from collections import Counter

from ultralytics import YOLO

results = YOLO("runs/detect/train/weights/best.pt").predict(
    source="videos/street.mp4", conf=0.25, save=True, stream=True, verbose=False
)

classes = Counter()
total_ms = frames = 0
save_dir = None
for r in results:
    frames += 1
    total_ms += sum(r.speed.values())
    classes.update(r.names[int(b.cls)] for b in r.boxes)
    save_dir = r.save_dir

print(f"frames processed: {frames}")
print(f"average speed:    {total_ms / frames:.1f} ms/frame")
print(f"output folder:    {save_dir}")
for name, n in classes.most_common():
    print(f"  {name:15s} {n}")
