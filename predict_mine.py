from ultralytics import YOLO

# Run your own trained model (after train.py finishes).
model = YOLO("runs/detect/train/weights/best.pt")
model.predict(source="https://ultralytics.com/images/zidane.jpg", save=True)
