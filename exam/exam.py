r"""
EXAM PROGRAM - fly the Tello by camera, recognise the teammate, save the photo.

Flow (see FLOW.md):
  1. Load our trained model (before connecting, so no time is lost later).
  2. Connect to the Tello, start the live video.
  3. Every frame: run YOLO, draw boxes with "name confidence".
  4. Pilot flies with the keyboard while watching this window.
  5. When the SAME teammate is seen in STABLE_FRAMES frames in a row
     (confidence >= MIN_CONF), keep watching for BEST_WINDOW seconds and
     save the frame with the HIGHEST RAW confidence.
  6. Check the saved photo opens correctly, show "SAVED OK" on screen.
  7. Pilot lands (l). ESC = land and quit.

The confidence shown and saved is EXACTLY what YOLO outputs (exam rule:
never enlarge, add to, or fix the number).

Keys (click the video window first):
  t take off | l land | w/s forward/back | a/d left/right | r/f up/down
  q/e turn | SPACE hover | p save photo now (manual backup)
  x EMERGENCY motors off (only if about to hit a person) | ESC land + quit

Run (from the UAV folder):
  .venv\Scripts\python.exe exam\exam.py
"""
import time
from pathlib import Path

import cv2
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO

# ---------------- SETTINGS ----------------
HERE = Path(__file__).parent
MODEL_PATH = HERE / "runs" / "team" / "weights" / "best.pt"
PHOTO_DIR = HERE / "exam_photos"
SPEED = 30            # flying speed (10-100)
MIN_BATTERY = 20      # no takeoff below this %
HOLD_TIME = 0.15      # seconds one key press keeps moving
MIN_CONF = 0.50       # ignore detections below this confidence
STABLE_FRAMES = 5     # same name this many frames in a row before saving
BEST_WINDOW = 1.0     # seconds to look for the clearest frame, then save
# ------------------------------------------

MOVES = {
    ord("w"): (0, SPEED, 0, 0),   ord("s"): (0, -SPEED, 0, 0),
    ord("a"): (-SPEED, 0, 0, 0),  ord("d"): (SPEED, 0, 0, 0),
    ord("r"): (0, 0, SPEED, 0),   ord("f"): (0, 0, -SPEED, 0),
    ord("q"): (0, 0, 0, -SPEED),  ord("e"): (0, 0, 0, SPEED),
}


def best_detection(result):
    """Return (name, raw_confidence, box) of the most confident detection, or None."""
    if result.boxes is None or len(result.boxes) == 0:
        return None
    i = int(result.boxes.conf.argmax())
    conf = float(result.boxes.conf[i])          # RAW confidence from YOLO
    name = result.names[int(result.boxes.cls[i])]
    box = [int(v) for v in result.boxes.xyxy[i].tolist()]
    return name, conf, box


def draw_detection(img, det, color=(0, 200, 0)):
    """Draw the box and 'name 0.912' label (3 decimals of the raw value)."""
    name, conf, (x1, y1, x2, y2) = det
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    label = f"{name} {conf:.3f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(img, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
    cv2.putText(img, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def save_photo(clean_frame, det):
    """Draw the result on a copy, save it, and check the file opens."""
    img = clean_frame.copy()
    draw_detection(img, det)
    PHOTO_DIR.mkdir(exist_ok=True)
    name, conf, _ = det
    path = PHOTO_DIR / f"{name}_{conf:.3f}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    ok = cv2.imwrite(str(path), img)
    ok = ok and cv2.imread(str(path)) is not None   # exam rule: photo must open
    return path, ok


# 1) Load the model first and warm it up (first prediction is always slow).
print("Loading model:", MODEL_PATH)
model = YOLO(str(MODEL_PATH))
model.predict(np.zeros((720, 960, 3), dtype=np.uint8), verbose=False)
print("Model ready. Classes:", model.names)

tello = Tello()
flying = False

try:
    # 2) Connect and start video.
    tello.connect()
    print("Battery:", tello.get_battery(), "%")
    tello.streamon()
    frame_reader = tello.get_frame_read()
    time.sleep(2)

    current_move, last_key_time = (0, 0, 0, 0), 0
    status = "LANDED - press t to take off"
    streak_name, streak_count = None, 0      # for the "same name N frames" check
    window_start, window_best = None, None   # best frame during BEST_WINDOW
    saved_text = ""

    while True:
        frame = frame_reader.frame
        if frame is None:
            continue
        clean = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)   # RGB -> BGR

        # 3) Recognise: run our model on this frame.
        result = model.predict(clean, conf=MIN_CONF, verbose=False)[0]
        det = best_detection(result)

        # 5) Stable-name check, then pick the clearest frame.
        if det and det[0] == streak_name:
            streak_count += 1
        else:
            streak_name, streak_count = (det[0] if det else None), (1 if det else 0)
            window_start, window_best = None, None

        if streak_count >= STABLE_FRAMES and not saved_text:
            if window_start is None:
                window_start = time.time()
            if window_best is None or det[1] > window_best[1][1]:
                window_best = (clean.copy(), det)       # keep the raw best
            if time.time() - window_start >= BEST_WINDOW:
                path, ok = save_photo(*window_best)
                saved_text = f"SAVED OK: {path.name}" if ok else "SAVE FAILED - press p"
                print(saved_text)

        # Draw the live view.
        view = clean.copy()
        if det:
            draw_detection(view, det)
        cv2.putText(view, f"Battery {tello.get_battery()}%  Height {tello.get_height()}cm",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(view, status, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if saved_text:
            cv2.putText(view, saved_text, (10, view.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if "OK" in saved_text else (0, 0, 255), 2)
        cv2.imshow("EXAM - Tello", view)

        # 4) Keyboard flying.
        key = cv2.waitKey(1) & 0xFF
        if key == ord("t") and not flying:
            if tello.get_battery() < MIN_BATTERY:
                status = f"Battery too low (< {MIN_BATTERY}%)"
            else:
                tello.takeoff()
                flying, status = True, "FLYING"
        elif key == ord("l") and flying:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
            flying, status = False, "LANDED"
        elif key == ord("x"):
            tello.emergency()
            flying, status = False, "EMERGENCY STOP"
        elif key == 27:  # ESC
            break
        elif key == ord(" "):
            current_move = (0, 0, 0, 0)
        elif key == ord("p"):                       # manual backup save
            if det:
                path, ok = save_photo(clean, det)
                saved_text = f"SAVED OK: {path.name}" if ok else "SAVE FAILED"
            else:
                saved_text = ""
                status = "Nothing recognised yet - get closer"
        elif key in MOVES:
            current_move, last_key_time = MOVES[key], time.time()

        if time.time() - last_key_time > HOLD_TIME:
            current_move = (0, 0, 0, 0)               # no key -> hover
        if flying:
            tello.send_rc_control(*current_move)      # also keeps the drone awake

finally:
    # 7) Safety: always try to land and close cleanly.
    try:
        if flying:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
    except Exception:
        pass
    try:
        tello.streamoff()
    except Exception:
        pass
    tello.end()
    cv2.destroyAllWindows()
