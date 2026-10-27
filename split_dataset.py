"""Move my-dataset/images/*.jpg into images/train and images/val (80/20),
carrying each image's .txt from labels/ into the matching labels/ folder."""

import random
import shutil
from pathlib import Path

ROOT = Path("my-dataset")
VAL_FRACTION = 0.2
SEED = 0  # fixed so re-running gives the same split

images = sorted((ROOT / "images").glob("*.jpg"))
if not images:
    raise SystemExit("No images to split - already split, or run extract_frames.py")

random.Random(SEED).shuffle(images)
n_val = round(len(images) * VAL_FRACTION)
splits = {"val": images[:n_val], "train": images[n_val:]}

missing = []
for split, files in splits.items():
    for sub in ("images", "labels"):
        (ROOT / sub / split).mkdir(parents=True, exist_ok=True)
    for img in files:
        shutil.move(img, ROOT / "images" / split / img.name)
        txt = ROOT / "labels" / f"{img.stem}.txt"
        if txt.exists():
            shutil.move(txt, ROOT / "labels" / split / txt.name)
        else:
            missing.append(txt.name)

for split in splits:
    n_img = len(list((ROOT / "images" / split).glob("*.jpg")))
    n_txt = len(list((ROOT / "labels" / split).glob("*.txt")))
    print(f"{split:5s}: {n_img} images, {n_txt} labels")
if missing:
    print(f"WARNING missing labels: {missing}")

# No image may end up in both splits, and every image needs its label.
train = {p.stem for p in (ROOT / "images" / "train").glob("*.jpg")}
val = {p.stem for p in (ROOT / "images" / "val").glob("*.jpg")}
assert not train & val, f"image in both splits: {train & val}"
assert not missing, "some images lost their label file"
print("checks passed: no overlap, every image has a label")
