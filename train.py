from ultralytics import YOLO

# Train YOLO26 on the chair-table dataset (2 classes: 0 = chair, 1 = table).
# 120 train photos, 30 val photos. Results go to runs/detect/train.
if __name__ == "__main__":  # required on Windows
    model = YOLO("yolo26n.pt")
    model.train(data="chair-table/data.yaml", epochs=50, imgsz=640)
