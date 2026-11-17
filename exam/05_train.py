r"""
STEP 5 - Train YOLO26 to recognise your teammates (no drone needed).

Uses the dataset made by 04_make_dataset.py.
Result: exam\runs\team\weights\best.pt  (this is the model used in exam.py)

On a laptop CPU this takes a while (roughly 20-60 min). On Google Colab's
free GPU it takes a few minutes: upload exam\team_dataset, run the same code.

Run (from the UAV folder):
  .venv\Scripts\python.exe exam\05_train.py
"""
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).parent

if __name__ == "__main__":  # required on Windows
    model = YOLO("yolo26n.pt")             # start from the pretrained model
    model.train(
        data=str(HERE / "team_dataset" / "data.yaml"),
        epochs=60,                         # passes over all photos
        imgsz=640,                         # photo size the model sees
        patience=20,                       # stop early if no improvement for 20 epochs
        project=str(HERE / "runs"),
        name="team",
        exist_ok=True,
    )
    print("Done. Model saved to exam\\runs\\team\\weights\\best.pt")
