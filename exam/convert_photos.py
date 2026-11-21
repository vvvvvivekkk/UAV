r"""
STEP A - Convert phone photos (HEIC / JPG / PNG) into upright JPGs.

Reads  exam\dataset_raw\Train\<name>\, \Test\<name>\, \background\
Writes exam\dataset_jpg\train\<name>\, \test\<name>\, \background\  (lowercase names)
Big photos are shrunk to max 1280 px. Already-converted photos are skipped.

Needs once:  .venv2\Scripts\python.exe -m pip install pillow-heif
Run (from the UAV folder):
  .venv2\Scripts\python.exe exam\convert_photos.py
"""
from pathlib import Path

from PIL import Image, ImageOps
import pillow_heif

pillow_heif.register_heif_opener()      # lets Pillow open iPhone .HEIC files

HERE = Path(__file__).parent
SRC = HERE / "dataset_raw"
DST = HERE / "dataset_jpg"

done = skipped = 0
for f in sorted(SRC.rglob("*")):
    if f.suffix.lower() not in (".heic", ".jpg", ".jpeg", ".png"):
        continue
    parts = [p.lower() for p in f.relative_to(SRC).parts[:-1]]   # Train/Vivek -> train/vivek
    out = DST.joinpath(*parts, f.stem + ".jpg")
    if out.exists():
        skipped += 1
        continue
    out.parent.mkdir(parents=True, exist_ok=True)
    img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")  # turn upright
    img.thumbnail((1280, 1280))
    img.save(out, quality=90)
    done += 1

print(f"Converted {done} photos, skipped {skipped} already done.")
for d in sorted(p for p in DST.rglob("*") if p.is_dir()):
    n = len(list(d.glob("*.jpg")))
    if n:
        print(f"  {d.relative_to(DST)}: {n}")
