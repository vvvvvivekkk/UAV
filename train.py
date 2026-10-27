from ultralytics import YOLO

# Train YOLO26 on your own labelled photos in my-dataset/.
# Put photos in my-dataset/images/train and /val, and their .txt labels in
# my-dataset/labels/train and /val first (see README).
if __name__ == "__main__":  # required on Windows
    model = YOLO("yolo26n.pt")
    # epochs=10, batch=8 for a quick CPU test. Use epochs=50+ for real training.
    model.train(data="my-dataset/data.yaml", epochs=10, imgsz=640, batch=8)
