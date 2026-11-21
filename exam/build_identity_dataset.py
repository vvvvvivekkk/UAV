r"""
STEP B - Build the identity dataset: crops of each person, sorted into
vivek / praney / other.

  - In vivek / praney photos the BIGGEST person is that member.
    Small people far away in those photos become "other".
  - Every person in background photos (classmates) becomes "other".
  - Crop = head + upper body (faces, hair, glasses matter more than clothes).
  - train photos -> 85% train / 15% val.  test photos -> test (never trained on).
  - Members have far fewer photos than "other", so their training crops are
    repeated (vivek x4, praney x12) so the model doesn't just say "other".

Output: exam\identity_dataset\{train,val,test}\{vivek,praney,other}\*.jpg
CHECK the crops afterwards (open the folders): if a classmate ended up in
vivek\ or praney\, move that crop into other\.

Run (from the UAV folder):
  .venv2\Scripts\python.exe exam\build_identity_dataset.py
"""
import random
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

# ---------------- SETTINGS ----------------
HERE = Path(__file__).parent
SRC = HERE / "dataset_jpg"
OUT = HERE / "identity_dataset"
MEMBERS = ["vivek", "praney"]
REPEAT = {"vivek": 4, "praney": 12}   # balance small classes (adjust if you add photos)
VAL_SHARE = 0.15
# ------------------------------------------

person_model = YOLO(str(HERE.parent / "yolo26n.pt"))
random.seed(0)
if OUT.exists():
    shutil.rmtree(OUT)


def upper_crop(img, box):
    """Head + upper body: top ~55% of the person box, a bit of padding."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    ch = max(h * 0.55, w * 1.1)
    X1, Y1 = max(0, x1 - 0.08 * w), max(0, y1 - 0.06 * h)
    X2, Y2 = min(img.shape[1], x2 + 0.08 * w), min(img.shape[0], y1 + ch)
    return img[int(Y1):int(Y2), int(X1):int(X2)]


def find_people(img):
    """All person boxes, biggest first."""
    r = person_model.predict(img, classes=[0], conf=0.35, verbose=False)[0]
    return sorted(r.boxes.xyxy.tolist(), key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))


def overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def save(crop, split, cls, name):
    if crop.shape[0] < 40 or crop.shape[1] < 30:
        return
    d = OUT / split / cls
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / name), crop)


# 1) Team member photos
for src_split in ["train", "test"]:
    for member in MEMBERS:
        for f in sorted((SRC / src_split / member).glob("*.jpg")):
            img = cv2.imread(str(f))
            people = find_people(img)
            if not people:
                print("no person found:", f.name)
                continue
            split = "test" if src_split == "test" else ("val" if random.random() < VAL_SHARE else "train")
            main = people[0]
            save(upper_crop(img, main), split, member, f.stem + ".jpg")
            main_area = (main[2] - main[0]) * (main[3] - main[1])
            for j, p in enumerate(people[1:]):
                if (p[2] - p[0]) * (p[3] - p[1]) < 0.35 * main_area and overlap(p, main) < 0.05:
                    save(upper_crop(img, p), split, "other", f"{f.stem}_o{j}.jpg")

# 2) Background photos: everyone is "other" (70% train / 15% val / 15% test)
bg = sorted((SRC / "background").glob("*.jpg"))
random.shuffle(bg)
for i, f in enumerate(bg):
    split = "train" if i < 0.7 * len(bg) else ("val" if i < 0.85 * len(bg) else "test")
    img = cv2.imread(str(f))
    for j, p in enumerate(find_people(img)):
        save(upper_crop(img, p), split, "other", f"{f.stem}_{j}.jpg")

# 3) Repeat member training crops to balance the classes
for member, times in REPEAT.items():
    d = OUT / "train" / member
    originals = sorted(d.glob("*.jpg"))
    for k in range(1, times):
        for f in originals:
            shutil.copy(f, d / f"{f.stem}_dup{k}.jpg")

for split in ["train", "val", "test"]:
    counts = {c.name: len(list(c.glob("*.jpg"))) for c in sorted((OUT / split).iterdir())}
    print(split, counts)
print("Dataset ready:", OUT)
