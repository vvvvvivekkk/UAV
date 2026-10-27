from ultralytics import YOLO

# Train YOLO26 on your own labelled photos in my-dataset/.
# Put photos in my-dataset/images/train and /val, and their .txt labels in
# my-dataset/labels/train and /val first (see README).
if __name__ == "__main__":  # required on Windows
    model = YOLO("yolo26n.pt")
    model.train(data="my-dataset/data.yaml", epochs=50, imgsz=640)
