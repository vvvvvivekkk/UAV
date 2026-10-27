"""Write PSEUDO-labels for my-dataset/images/ using yolo26n.pt.

These are a model's guesses, NOT human labels. Good enough to test that the
training pipeline runs; not good enough to judge whether a model is accurate.
For real work, draw the boxes yourself in X-AnyLabeling and export "YOLO HBB".
"""

from pathlib import Path

from ultralytics import YOLO

IMAGES = Path("my-dataset/images")
LABELS = Path("my-dataset/labels")
# COCO id -> our classes.txt id. classes.txt is: person=0, bus=1
REMAP = {0: 0, 5: 1}

LABELS.mkdir(parents=True, exist_ok=True)
images = sorted(IMAGES.glob("*.jpg"))
if not images:
    raise SystemExit(f"No images in {IMAGES} - run extract_frames.py first")

results = YOLO("yolo26n.pt").predict(
    source=images, conf=0.25, classes=list(REMAP), stream=True, verbose=False
)

boxes_written = 0
for r in results:
    lines = []
    for box in r.boxes:
        cls = REMAP[int(box.cls)]
        # xywhn = centre-x, centre-y, width, height, already scaled to 0-1
        x, y, w, h = box.xywhn[0].tolist()
        lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    boxes_written += len(lines)
    # An empty .txt is valid YOLO: "this image contains no objects".
    (LABELS / (Path(r.path).stem + ".txt")).write_text("\n".join(lines) + "\n")

print(f"wrote {len(images)} label files, {boxes_written} boxes total, to {LABELS}")
print("REMINDER: these are pseudo-labels from a model, not human labels.")
