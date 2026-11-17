r"""
STEP 3 - Collect training photos of each teammate with the Tello camera.
The drone does NOT need to fly: put it on a table or hold it at chest height.

Click on the video window first, then:
  1 / 2     = choose who is in front of the camera (1 = vivek, 2 = praney)
  0         = background (empty room, nobody in view)
  c         = save ONE photo
  v (hold)  = save a photo every 0.3 seconds (burst)
  ESC       = quit

Photos are saved WITHOUT any text on them, here:
  exam\dataset_raw\<name>\<name>_<time>.jpg

Goal: about 100 photos per teammate (1-2 m away, turn head a little,
step closer/farther) + about 30 background photos.

Run (from the UAV folder):
  .venv\Scripts\python.exe exam\03_collect_photos.py
"""
import time
from pathlib import Path

import cv2
from djitellopy import Tello

# ---------------- SETTINGS ----------------
MEMBERS = ["vivek", "praney"]                 # our team (key 1 = vivek, key 2 = praney)
BURST_GAP = 0.3                               # seconds between burst photos
SAVE_DIR = Path(__file__).parent / "dataset_raw"
# ------------------------------------------

# Key -> folder name
CHOICES = {ord(str(i + 1)): name for i, name in enumerate(MEMBERS)}
CHOICES[ord("0")] = "background"

def count_photos(name):
    folder = SAVE_DIR / name
    return len(list(folder.glob("*.jpg"))) if folder.exists() else 0

def save_photo(frame_bgr, name):
    folder = SAVE_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}_{time.strftime('%H%M%S')}_{int(time.time()*1000)%1000:03d}.jpg"
    cv2.imwrite(str(path), frame_bgr)
    return path

tello = Tello()

try:
    tello.connect()
    print("Battery:", tello.get_battery(), "%")
    tello.streamon()
    frame_reader = tello.get_frame_read()
    time.sleep(2)

    current = MEMBERS[0]
    last_burst = 0

    while True:
        frame = frame_reader.frame
        if frame is None:
            continue
        clean = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # this copy gets saved
        view = clean.copy()                              # this copy gets text

        # Draw the current name big, plus a counter for everyone.
        cv2.putText(view, f"NOW: {current}", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        y = 90
        for name in MEMBERS + ["background"]:
            cv2.putText(view, f"{name}: {count_photos(name)} photos", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30
        cv2.putText(view, "1 vivek | 2 praney | 0 empty room | c photo | hold v burst | ESC quit",
                    (10, view.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        cv2.imshow("Collect photos", view)

        key = cv2.waitKey(1) & 0xFF
        if key in CHOICES:
            current = CHOICES[key]
        elif key == ord("c"):
            print("Saved", save_photo(clean, current))
        elif key == ord("v") and time.time() - last_burst > BURST_GAP:
            save_photo(clean, current)
            last_burst = time.time()
        elif key == 27:  # ESC
            break

    print("Final counts:", {n: count_photos(n) for n in MEMBERS + ["background"]})

finally:
    try:
        tello.streamoff()
    except Exception:
        pass
    tello.end()
    cv2.destroyAllWindows()
