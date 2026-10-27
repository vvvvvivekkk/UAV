from collections import Counter

from ultralytics import YOLO

# stream=True yields one result per frame instead of building a 480-item list
# in RAM. save=True still writes the annotated .mp4.
results = YOLO("yolo26n.pt").predict(
    source="videos/street.mp4", conf=0.25, save=True, stream=True, verbose=False
)

classes = Counter()
total_ms = frames = 0
save_dir = None
for r in results:
    frames += 1
    total_ms += sum(r.speed.values())  # preprocess + inference + postprocess
    classes.update(r.names[int(b.cls)] for b in r.boxes)
    save_dir = r.save_dir

print(f"frames processed: {frames}")
print(f"average speed:    {total_ms / frames:.1f} ms/frame")
print(f"output folder:    {save_dir}")
print("class totals (summed over all frames):")
for name, n in classes.most_common():
    print(f"  {name:15s} {n}")
