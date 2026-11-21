r"""
IDENTIFY - find every person in a photo or video and label them:
    "vivek 0.93"  /  "praney 0.88"  /  "other person"

How it works:
  1. face_detector.onnx (OpenCV YuNet) finds every FACE -> a box.
  2. identity.pt (our trained YOLO26 classifier) looks at each face (+ hair, glasses)
     and says vivek / praney / other, with a RAW confidence (this is the number shown).
     If it isn't sure enough (below MIN_NAME_CONF) -> "other person".
  3. Double check: face_recognizer.onnx (OpenCV SFace) compares the face with the
     saved photos of that member (models/face_gallery.npz). If it doesn't look
     enough like them (similarity below MIN_SIMILARITY) -> "other person".
     This stops strangers from getting a member's name.
  4. yolo26n.pt finds every PERSON. A person whose face can't be seen
     (turned away, too far) gets a grey "other person" box.
We use the face, not the clothes: with clothes the model called a classmate
in a light-blue hoodie "praney".

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
import numpy as np
from ultralytics import YOLO

# ---------------- SETTINGS ----------------
HERE = Path(__file__).parent
PERSON_MODEL = HERE.parent / "yolo26n.pt"              # finds people
FACE_MODEL = HERE / "models" / "face_detector.onnx"    # finds faces
RECOGNIZER = HERE / "models" / "face_recognizer.onnx"  # face fingerprints (double check)
GALLERY = HERE / "models" / "face_gallery.npz"         # fingerprints of vivek / praney photos
IDENTITY_MODEL = HERE / "models" / "identity.pt"       # vivek / praney / other
OUT_DIR = HERE / "identify_results"
PERSON_CONF = 0.40     # how sure step 3 must be that it's a person
FACE_CONF = 0.60       # how sure step 1 must be that it's a face
MIN_FACE = 20          # faces smaller than this (pixels) are too small to name
MIN_NAME_CONF = 0.60   # below this, a "vivek"/"praney" guess is shown as "other person"
MIN_SIMILARITY = 0.50  # double check: face must be at least this similar to the member's photos
MAX_SIZE = 1280        # shrink 4K phone videos/photos to this (faster, smaller files)
MEMBERS = ("vivek", "praney")
COLORS = {"vivek": (0, 200, 0), "praney": (255, 140, 0), "other": (160, 160, 160)}  # BGR
# ------------------------------------------

person_model = YOLO(str(PERSON_MODEL))
identity_model = YOLO(str(IDENTITY_MODEL))
face_detector = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), FACE_CONF)
recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER), "")
gallery = dict(np.load(GALLERY))


def find_faces(img):
    """All faces. Each face: x, y, w, h, then eye/nose/mouth points."""
    face_detector.setInputSize((img.shape[1], img.shape[0]))
    _, faces = face_detector.detect(img)
    if faces is None:
        return []
    return [f for f in faces if min(f[2], f[3]) >= MIN_FACE]


def similarity(img, face, member):
    """How much this face looks like the saved photos of member (best match, 0..1)."""
    e = recognizer.feature(recognizer.alignCrop(img, face)).flatten()
    e = e / np.linalg.norm(e)
    return float((gallery[member] @ e).max())


def face_crop(img, face):
    """Face + hair + glasses (same crop as used in training)."""
    x, y, w, h = face[:4]
    X1, Y1 = max(0, x - 0.35 * w), max(0, y - 0.45 * h)
    X2, Y2 = min(img.shape[1], x + 1.35 * w), min(img.shape[0], y + 1.25 * h)
    return img[int(Y1):int(Y2), int(X1):int(X2)], [int(X1), int(Y1), int(X2), int(Y2)]


def inside(face, box):
    """Is the centre of the face inside the person box?"""
    cx, cy = face[0] + face[2] / 2, face[1] + face[3] / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def identify(frame):
    """Return a list of (label, raw_conf or None, box) for every person in the frame."""
    found = []
    faces = find_faces(frame)
    for face in faces:
        crop, box = face_crop(frame, face)
        r = identity_model.predict(crop, verbose=False)[0]
        name = r.names[int(r.probs.top1)]
        conf = float(r.probs.top1conf)            # RAW confidence from the model
        found.append([name, conf, box, face])

    # Each team member can appear only once: keep the most confident one.
    for member in MEMBERS:
        hits = [f for f in found if f[0] == member]
        for f in sorted(hits, key=lambda f: -f[1])[1:]:
            f[0] = "other"

    results = []
    for name, conf, box, face in found:
        if name in MEMBERS and conf >= MIN_NAME_CONF and similarity(frame, face, name) >= MIN_SIMILARITY:
            results.append((name, conf, box))
        else:
            results.append(("other person", None, box))

    # People whose face we can't see -> "other person" box on the body
    people = person_model.predict(frame, classes=[0], conf=PERSON_CONF, verbose=False)[0]
    for box in people.boxes.xyxy.tolist():
        if not any(inside(face, box) for face in faces):
            results.append(("other person", None, [int(v) for v in box]))
    return results


def shrink(frame):
    """Make big (e.g. 4K) frames smaller so everything runs faster."""
    h, w = frame.shape[:2]
    s = MAX_SIZE / max(h, w)
    return cv2.resize(frame, (int(w * s), int(h * s))) if s < 1 else frame


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
        img = shrink(img)
        res = identify(img)
        cv2.imwrite(str(OUT_DIR / f"{p.stem}_labeled.jpg"), draw(img, res))
        names = ", ".join(f"{l} {c:.2f}" if c is not None else l for l, c, _ in res) or "nobody"
        print(f"{p.name}: {names}")
    print("Saved labelled photos in", OUT_DIR)


def run_video(path):
    OUT_DIR.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = OUT_DIR / f"{path.stem}_labeled.mp4"
    writer = None
    stats, n, t0 = {}, 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = shrink(frame)
        if writer is None:   # create the output video with the (shrunk) frame size
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
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
    if writer is not None:
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
