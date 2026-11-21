r"""
STEP C - Train the identity model (YOLO26 classifier: vivek / praney / other).

Uses exam\identity_dataset (from build_identity_dataset.py).
Saves the model to exam\models\identity.pt (used by identify.py and exam.py)
and prints the accuracy on the TEST crops (photos never trained on).

GPU (RTX 4060): a few minutes.  CPU: ~15-30 minutes.
Run (from the UAV folder):
  .venv2\Scripts\python.exe exam\train_identity.py
"""
import shutil
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).parent

if __name__ == "__main__":  # required on Windows
    model = YOLO("yolo26n-cls.pt")          # pretrained classifier (downloads once)
    model.train(
        data=str(HERE / "identity_dataset"),
        epochs=40,
        imgsz=224,
        batch=32,
        patience=15,                        # stop early if it stops improving
        project=str(HERE / "runs"),
        name="identity",
        exist_ok=True,
    )
    best = HERE / "runs" / "identity" / "weights" / "best.pt"
    (HERE / "models").mkdir(exist_ok=True)
    shutil.copy(best, HERE / "models" / "identity.pt")

    result = YOLO(str(HERE / "models" / "identity.pt")).val(
        data=str(HERE / "identity_dataset"), split="test", imgsz=224)
    print(f"\nTEST accuracy: {result.top1:.3f}")
    print("Model saved to exam\\models\\identity.pt")
