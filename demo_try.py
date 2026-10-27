from ultralytics import YOLO

model = YOLO("yolo26n.pt")


def run(label, **kw):
    r = model.predict(save=True, verbose=False, **kw)[0]
    names = [r.names[int(b.cls)] for b in r.boxes]
    print(f"{label}: {len(r.boxes)} boxes -> {names}")
    print(f"  saved to {r.save_dir}")


ZIDANE = "https://ultralytics.com/images/zidane.jpg"
BUS = "https://ultralytics.com/images/bus.jpg"

# Same image, two thresholds: shows what conf actually filters out.
run("zidane conf=0.25", source=ZIDANE, conf=0.25)
run("zidane conf=0.50", source=ZIDANE, conf=0.50)

# Same bus photo as demo.py, but only COCO class 0 (person).
run("bus classes=[0]", source=BUS, conf=0.25, classes=[0])

# 0.50 doesn't drop anything here (lowest score is tie=0.584).
# 0.60 does — proof that conf is just a cutoff on the score.
run("zidane conf=0.60", source=ZIDANE, conf=0.60)
