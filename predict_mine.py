from ultralytics import YOLO

# Run YOUR trained chair/table model (best.pt) on a val photo it never trained on.
model = YOLO("runs/detect/train/weights/best.pt")
model.predict(source="chair-table/images/val/138.jpg", save=True)
