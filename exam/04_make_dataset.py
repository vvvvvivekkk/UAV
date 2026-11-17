r"""
STEP 4 - Turn your collected photos into a YOLO dataset (no drone needed).

What it does:
  1. For every photo in exam\dataset_raw\<name>\ it finds the PERSON with
     the pretrained yolo26n.pt (it already knows "person") and writes a YOLO label:
        <class number> <x_center> <y_center> <width> <height>   (all 0-1)
     The class number comes from the folder name (MEMBERS order).
  2. Background photos get an EMPTY label (= "nobody here").
  3. Splits photos 80% train / 20% val into exam\team_dataset\.
  4. Writes exam\team_dataset\data.yaml.
  5. Lists photos where no person was found -> label those yourself in
     X-AnyLabeling (or just skip them).

Note: the model learns each teammate from face, hair AND clothes, so wear the
SAME clothes on exam day as when you collected photos (or collect again).

Check the boxes afterwards: open exam\team_dataset in X-AnyLabeling and fix
any wrong box. YOU are responsible for the labels being correct.

Run (from the UAV folder):
  .venv2\Scripts\python.exe exam\04_make_dataset.py
"""
import random
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

# ---------------- SETTINGS ----------------
MEMBERS = ["vivek", "praney"]                 # SAME order as in 03_collect_photos.py
VAL_SHARE = 0.2                               # 20% of photos go to validation
MIN_PERSON_CONF = 0.4                         # how sure yolo26n must be it's a person
HERE = Path(__file__).parent
RAW = HERE / "dataset_raw"
OUT = HERE / "team_dataset"
# ------------------------------------------

# The pretrained COCO model: class 0 = "person"
person_finder = YOLO(str(HERE.parent / "yolo26n.pt"))

def find_person(img):
    """Return the biggest person box as (x1, y1, x2, y2) in pixels, or None."""
    r = person_finder.predict(img, classes=[0], conf=MIN_PERSON_CONF, verbose=False)[0]
    if len(r.boxes) == 0:
        return None
    boxes = r.boxes.xyxy.tolist()
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

def to_yolo(box, img_w, img_h):
    """Pixel box -> YOLO numbers (centre x, centre y, width, height), each 0-1."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h)

# Start fresh each time
if OUT.exists():
    shutil.rmtree(OUT)
for split in ["train", "val"]:
    (OUT / "images" / split).mkdir(parents=True)
    (OUT / "labels" / split).mkdir(parents=True)

random.seed(0)
missing, counts = [], {}
for class_id, name in enumerate(MEMBERS + ["background"]):
    photos = sorted((RAW / name).glob("*.jpg")) if (RAW / name).exists() else []
    random.shuffle(photos)
    n_val = max(1, int(len(photos) * VAL_SHARE)) if photos else 0
    counts[name] = 0
    for i, photo in enumerate(photos):
        split = "val" if i < n_val else "train"
        img = cv2.imread(str(photo))
        if img is None:
            continue
        if name == "background":
            label = ""                       # empty label = no person
        else:
            box = find_person(img)
            if box is None:
                missing.append(str(photo))   # needs a manual box
                continue
            cx, cy, w, h = to_yolo(box, img.shape[1], img.shape[0])
            label = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
        shutil.copy(photo, OUT / "images" / split / photo.name)
        (OUT / "labels" / split / (photo.stem + ".txt")).write_text(label)
        counts[name] += 1

# data.yaml tells YOLO where the photos are and the class names
names = "\n".join(f"  {i}: {n}" for i, n in enumerate(MEMBERS))
(OUT / "data.yaml").write_text(f"path: {OUT.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
(OUT / "classes.txt").write_text("\n".join(MEMBERS) + "\n")

print("Photos used per class:", counts)
print(f"No person found in {len(missing)} photos (not used). First few:")
for m in missing[:10]:
    print("  ", m)
print("Dataset ready:", OUT)
