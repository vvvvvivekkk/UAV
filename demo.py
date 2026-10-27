from ultralytics import YOLO

# Load the smallest YOLO26 model. It downloads automatically on the first run.
model = YOLO("yolo26n.pt")

# Find objects in a sample photo.
# conf=0.25 hides boxes under 25% confidence; save=True writes the result image.
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    conf=0.25,
    save=True,
)

# results[0].boxes has one entry per detected object.
print("Detections:", len(results[0].boxes))
print("Open the image in runs/detect/predict to see the boxes.")
