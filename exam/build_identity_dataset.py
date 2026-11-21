r"""
STEP B - Build the identity dataset: FACE crops of each person, sorted into
vivek / praney / other.

Why faces (not the whole upper body)? With upper-body crops the model learned
"light-blue hoodie = praney" and named a classmate in a blue hoodie "praney".
A face crop (plus hair and glasses) makes the model learn the FACE instead of clothes.

  - Faces are found with OpenCV's YuNet face detector (models\face_detector.onnx).
  - It also saves a "face fingerprint" of every member training face
    (OpenCV SFace, models\face_recognizer.onnx) to models\face_gallery.npz.
    identify.py uses it as a double check: a face only gets a member's name if it
    also looks like that member's saved photos.
  - In vivek / praney photos the BIGGEST face is that member.
    Clearly smaller faces in those photos become "other".
  - Every face in background photos (classmates) becomes "other".
  - train photos -> 85% train / 15% val.  test photos -> test (never trained on).
  - Members have fewer crops than "other", so their training crops are repeated
    (see REPEAT) so the model doesn't just say "other".

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
import numpy as np

# ---------------- SETTINGS ----------------
HERE = Path(__file__).parent
SRC = HERE / "dataset_jpg"
OUT = HERE / "identity_dataset"
FACE_MODEL = HERE / "models" / "face_detector.onnx"
RECOGNIZER = HERE / "models" / "face_recognizer.onnx"
GALLERY = HERE / "models" / "face_gallery.npz"
MEMBERS = ["vivek", "praney"]
VAL_SHARE = 0.15
MIN_FACE = 24          # ignore faces smaller than this (pixels)
# ------------------------------------------

face_detector = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), 0.6)
recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER), "")
random.seed(0)
if OUT.exists():
    shutil.rmtree(OUT)


def find_faces(img):
    """All faces, biggest first. Each face: x, y, w, h, then eye/nose/mouth points."""
    face_detector.setInputSize((img.shape[1], img.shape[0]))
    _, faces = face_detector.detect(img)
    if faces is None:
        return []
    faces = [f for f in faces if min(f[2], f[3]) >= MIN_FACE]
    return sorted(faces, key=lambda f: -f[2] * f[3])


def fingerprint(img, face):
    """SFace 'face fingerprint' (128 numbers, length 1)."""
    e = recognizer.feature(recognizer.alignCrop(img, face)).flatten()
    return e / np.linalg.norm(e)


def face_crop(img, face):
    """Face + hair + glasses: the face box made bigger on every side."""
    x, y, w, h = face[:4]
    X1, Y1 = max(0, x - 0.35 * w), max(0, y - 0.45 * h)
    X2, Y2 = min(img.shape[1], x + 1.35 * w), min(img.shape[0], y + 1.25 * h)
    return img[int(Y1):int(Y2), int(X1):int(X2)]


def save(crop, split, cls, name):
    d = OUT / split / cls
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / name), crop)


# 1) Team member photos
gallery = {m: [] for m in MEMBERS}
for src_split in ["train", "test"]:
    for member in MEMBERS:
        for f in sorted((SRC / src_split / member).glob("*.jpg")):
            img = cv2.imread(str(f))
            faces = find_faces(img)
            if not faces:
                print("no face found:", f.name)
                continue
            split = "test" if src_split == "test" else ("val" if random.random() < VAL_SHARE else "train")
            main = faces[0]
            save(face_crop(img, main), split, member, f.stem + ".jpg")
            if src_split == "train":
                gallery[member].append(fingerprint(img, main))
            for j, face in enumerate(faces[1:]):
                if face[2] * face[3] < 0.5 * main[2] * main[3]:     # clearly smaller -> someone else
                    save(face_crop(img, face), split, "other", f"{f.stem}_o{j}.jpg")

np.savez(GALLERY, **{m: np.array(v) for m, v in gallery.items()})
print("Face fingerprints saved:", {m: len(v) for m, v in gallery.items()}, "->", GALLERY)

# 2) Background photos: every face is "other" (70% train / 15% val / 15% test)
bg = sorted((SRC / "background").glob("*.jpg"))
random.shuffle(bg)
for i, f in enumerate(bg):
    split = "train" if i < 0.7 * len(bg) else ("val" if i < 0.85 * len(bg) else "test")
    img = cv2.imread(str(f))
    for j, face in enumerate(find_faces(img)):
        save(face_crop(img, face), split, "other", f"{f.stem}_{j}.jpg")

# 3) Repeat member training crops so every class has about as many crops as "other"
n_other = len(list((OUT / "train" / "other").glob("*.jpg")))
for member in MEMBERS:
    d = OUT / "train" / member
    originals = sorted(d.glob("*.jpg"))
    times = max(1, round(n_other / max(1, len(originals))))
    for k in range(1, times):
        for f in originals:
            shutil.copy(f, d / f"{f.stem}_dup{k}.jpg")

for split in ["train", "val", "test"]:
    counts = {c.name: len(list(c.glob("*.jpg"))) for c in sorted((OUT / split).iterdir())}
    print(split, counts)
print("Dataset ready:", OUT)
