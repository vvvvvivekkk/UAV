r"""
AFTER THE EXAM - put everything to hand in into one zip file.

Includes: all code in exam\, README/FLOW, all models (models\identity.pt names faces,
face detector / recognizer / gallery in models\, yolo26n.pt finds people),
and every photo in exam\exam_photos\.

Run (from the UAV folder):
  .venv2\Scripts\python.exe exam\make_submission.py
"""
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
zip_path = HERE.parent / f"submission_{time.strftime('%Y%m%d_%H%M')}.zip"

files = list(HERE.glob("*.py")) + list(HERE.glob("*.md"))
files += [HERE / "models" / "identity.pt",
          HERE / "models" / "face_detector.onnx",
          HERE / "models" / "face_recognizer.onnx",
          HERE / "models" / "face_gallery.npz",
          HERE.parent / "yolo26n.pt"]
files += sorted((HERE / "exam_photos").glob("*.jpg"))

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        if f.exists():
            z.write(f, f.relative_to(HERE.parent))
            print("added", f.relative_to(HERE.parent))
        else:
            print("MISSING", f.relative_to(HERE.parent))
print("Submission file:", zip_path)
