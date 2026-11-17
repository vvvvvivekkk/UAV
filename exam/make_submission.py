r"""
AFTER THE EXAM - put everything to hand in into one zip file.

Includes: all code in exam\, README/FLOW, the model (best.pt), data.yaml,
classes.txt, and every photo in exam\exam_photos\.

Run (from the UAV folder):
  .venv\Scripts\python.exe exam\make_submission.py
"""
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
zip_path = HERE.parent / f"submission_{time.strftime('%Y%m%d_%H%M')}.zip"

files = list(HERE.glob("*.py")) + list(HERE.glob("*.md"))
files += [HERE / "runs" / "team" / "weights" / "best.pt",
          HERE / "team_dataset" / "data.yaml",
          HERE / "team_dataset" / "classes.txt"]
files += sorted((HERE / "exam_photos").glob("*.jpg"))

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        if f.exists():
            z.write(f, f.relative_to(HERE.parent))
            print("added", f.relative_to(HERE.parent))
        else:
            print("MISSING", f.relative_to(HERE.parent))
print("Submission file:", zip_path)
