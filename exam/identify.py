r"""
IDENTIFY - find every person in a photo or video and label them:
    "vivek 0.93"  /  "praney 0.88"  /  "other person"

How it works (2 steps, both YOLO26):
  1. yolo26n.pt (pretrained, knows "person") finds every person -> a box.
  2. identity.pt (our trained classifier) looks at the head + upper body
     inside each box and says vivek / praney / other, with a RAW confidence.
     If it isn't sure enough (below MIN_NAME_CONF) -> "other person".

Usage (from the UAV folder):
  .venv2\Scripts\python.exe exam\identify.py <photo | folder | video>
Examples:
  .venv2\Scripts\python.exe exam\identify.py exam\videos\test1.mp4
  .venv2\Scripts\python.exe exam\identify.py exam\dataset_jpg\test\praney

Results are saved in exam\identify_results\ (labelled photos / labelled video).
"""
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# ---------------- SETTINGS ----------------
HERE = Path(__file__).parent
PERSON_MODEL = HERE.parent / "yolo26n.pt"            # finds people
IDENTITY_MODEL = HERE / "models" / "identity.pt"     # vivek / praney / other
OUT_DIR = HERE / "identify_results"
PERSON_CONF = 0.40     # how sure step 1 must be that it's a person
MIN_NAME_CONF = 0.60   # below this, a "vivek"/"praney" guess is shown as "other person"
MEMBERS = ("vivek", "praney")
COLORS = {"vivek": (0, 200, 0), "praney": (255, 140, 0), "other": (160, 160, 160)}  # BGR
# ------------------------------------------

person_model = YOLO(str(PERSON_MODEL))
identity_model = YOLO(str(IDENTITY_MODEL))


def upper_crop(img, box):
    """Head + upper body part of a person box (same as used in training)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    ch = max(h * 0.55, w * 1.1)
    X1, Y1 = max(0, x1 - 0.08 * w), max(0, y1 - 0.06 * h)
    X2, Y2 = min(img.shape[1], x2 + 0.08 * w), min(img.shape[0], y1 + ch)
    return img[int(Y1):int(Y2), int(X1):int(X2)]


def identify(frame):
    """Return a list of (label, raw_conf or None, box) for every person in the frame."""
    people = person_model.predict(frame, classes=[0], conf=PERSON_CONF, verbose=False)[0]
    found = []
    for box in people.boxes.xyxy.tolist():
        crop = upper_crop(frame, box)
        if crop.shape[0] < 20 or crop.shape[1] < 15:
            continue
        r = identity_model.predict(crop, verbose=False)[0]
        name = r.names[int(r.probs.top1)]
        conf = float(r.probs.top1conf)            # RAW confidence from the model
        found.append([name, conf, [int(v) for v in box]])

    # Each team member can appear only once: keep the most confident one.
    for member in MEMBERS:
        hits = [f for f in found if f[0] == member]
        for f in sorted(hits, key=lambda f: -f[1])[1:]:
            f[0] = "other"

    results = []
    for name, conf, box in found:
        if name in MEMBERS and conf >= MIN_NAME_CONF:
            results.append((name, conf, box))
        else:
            results.append(("other person", None, box))
    return results


def draw(frame, results):
    for label, conf, (x1, y1, x2, y2) in results:
        color = COLORS.get(label, COLORS["other"])
        text = f"{label} {conf:.2f}" if conf is not None else label
        thick = 3 if conf is not None else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        y = max(y1, th + 10)
        cv2.rectangle(frame, (x1, y - th - 10), (x1 + tw + 8, y), color, -1)
        cv2.putText(frame, text, (x1 + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def run_images(paths):
    OUT_DIR.mkdir(exist_ok=True)
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        res = identify(img)
        cv2.imwrite(str(OUT_DIR / f"{p.stem}_labeled.jpg"), draw(img, res))
        names = ", ".join(f"{l} {c:.2f}" if c is not None else l for l, c, _ in res) or "nobody"
        print(f"{p.name}: {names}")
    print("Saved labelled photos in", OUT_DIR)


def run_video(path):
    OUT_DIR.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = OUT_DIR / f"{path.stem}_labeled.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    stats, n, t0 = {}, 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = identify(frame)
        writer.write(draw(frame, res))
        for label, conf, _ in res:
            s = stats.setdefault(label, [0, 0.0])
            s[0] += 1
            s[1] += conf or 0
        n += 1
        if n % 30 == 0:
            print(f"  frame {n}/{total}  ({n / (time.time() - t0):.1f} frames/s)")
    cap.release()
    writer.release()
    print(f"Done: {n} frames -> {out_path}")
    for label, (count, total_conf) in sorted(stats.items()):
        avg = f", average confidence {total_conf / count:.2f}" if label in MEMBERS else ""
        print(f"  {label}: seen in {count} frames{avg}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    src = Path(sys.argv[1])
    videos = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
    if src.is_dir():
        run_images(sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")))
    elif src.suffix.lower() in videos:
        run_video(src)
    else:
        run_images([src])
