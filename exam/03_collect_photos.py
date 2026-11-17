r"""
STEP 3 - Collect training photos with the Tello camera - you can do it ALONE.

Two ways:
  A) Drone on a table/chair (no flying): press 1 or 2, then g.
  B) Drone flying: t = take off, turn it to face where you will stand,
     press 1 or 2, then g, walk to your spot and move slowly.

Keys (click the video window first):
  1 = vivek   2 = praney   0 = empty room (background)
  g = AUTO-CAPTURE: waits 5 s (walk to your spot), then saves a photo every
      0.4 s for 25 s (about 60 photos). Press g again to stop early.
  c = save ONE photo now
  Flying: t take off | l land | w/s fwd/back | a/d left/right | r/f up/down
          q/e turn | SPACE hover | x EMERGENCY motors off | ESC land + quit

During auto-capture: stand 1-2 m in front of the drone, turn your head
left/right a little, step closer/farther, smile/no smile. Do 2 rounds each.

Photos are saved WITHOUT any text, to: exam\dataset_raw\<name>\

Run (from the UAV folder):
  .venv2\Scripts\python.exe exam\03_collect_photos.py
"""
import time
from pathlib import Path

import cv2
from djitellopy import Tello

# ---------------- SETTINGS ----------------
MEMBERS = ["vivek", "praney"]                 # our team (key 1 = vivek, key 2 = praney)
SAVE_DIR = Path(__file__).parent / "dataset_raw"
AUTO_DELAY = 5        # seconds to walk to your spot after pressing g
AUTO_SECONDS = 25     # how long auto-capture runs
AUTO_GAP = 0.4        # seconds between auto photos
SPEED = 25            # flying speed (slow, for positioning)
MIN_BATTERY = 20
HOLD_TIME = 0.15
# ------------------------------------------

CHOICES = {ord(str(i + 1)): name for i, name in enumerate(MEMBERS)}
CHOICES[ord("0")] = "background"
MOVES = {
    ord("w"): (0, SPEED, 0, 0),   ord("s"): (0, -SPEED, 0, 0),
    ord("a"): (-SPEED, 0, 0, 0),  ord("d"): (SPEED, 0, 0, 0),
    ord("r"): (0, 0, SPEED, 0),   ord("f"): (0, 0, -SPEED, 0),
    ord("q"): (0, 0, 0, -SPEED),  ord("e"): (0, 0, 0, SPEED),
}


def count_photos(name):
    folder = SAVE_DIR / name
    return len(list(folder.glob("*.jpg"))) if folder.exists() else 0


def save_photo(frame_bgr, name):
    folder = SAVE_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}_{time.strftime('%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"
    cv2.imwrite(str(path), frame_bgr)
    return path


tello = Tello()
flying = False

try:
    tello.connect()
    print("Battery:", tello.get_battery(), "%")
    tello.streamon()
    frame_reader = tello.get_frame_read()
    time.sleep(2)

    current = MEMBERS[0]
    auto_start = None          # time g was pressed (None = auto-capture off)
    last_auto = 0
    current_move, last_key_time = (0, 0, 0, 0), 0
    status = "LANDED"

    while True:
        frame = frame_reader.frame
        if frame is None:
            continue
        clean = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)   # this copy gets saved
        view = clean.copy()                              # this copy gets text
        now = time.time()

        # Auto-capture: wait AUTO_DELAY, then save every AUTO_GAP for AUTO_SECONDS
        auto_text = ""
        if auto_start is not None:
            t = now - auto_start
            if t < AUTO_DELAY:
                auto_text = f"GO TO YOUR SPOT: starting in {AUTO_DELAY - t:.0f}s"
            elif t < AUTO_DELAY + AUTO_SECONDS:
                auto_text = f"CAPTURING {current}: {AUTO_DELAY + AUTO_SECONDS - t:.0f}s left"
                if now - last_auto >= AUTO_GAP:
                    save_photo(clean, current)
                    last_auto = now
            else:
                auto_start = None
                print("Auto-capture done:", current, count_photos(current), "photos")

        # Text on the live view only
        cv2.putText(view, f"NOW: {current}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        y = 90
        for name in MEMBERS + ["background"]:
            cv2.putText(view, f"{name}: {count_photos(name)} photos", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30
        cv2.putText(view, f"Battery {tello.get_battery()}%  Temp {tello.get_temperature()}C  {status}",
                    (10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if auto_text:
            cv2.putText(view, auto_text, (10, view.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(view, "1 vivek | 2 praney | 0 room | g auto | c photo | t/l fly | ESC quit",
                    (10, view.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Collect photos", view)
        cv2.setWindowProperty("Collect photos", cv2.WND_PROP_TOPMOST, 1)  # keep window in front so it gets the keys

        key = cv2.waitKey(1) & 0xFF
        if key in CHOICES:
            current = CHOICES[key]
        elif key == ord("g"):
            auto_start = None if auto_start is not None else now   # toggle
        elif key == ord("c"):
            print("Saved", save_photo(clean, current))
        elif key == ord("t") and not flying:
            if tello.get_battery() < MIN_BATTERY:
                status = "BATTERY LOW"
            else:
                tello.takeoff()
                flying, status = True, "FLYING"
        elif key == ord("l") and flying:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
            flying, status = False, "LANDED"
        elif key == ord("x"):
            tello.emergency()
            flying, status = False, "EMERGENCY"
        elif key == ord(" "):
            current_move = (0, 0, 0, 0)
        elif key == 27:  # ESC
            break
        elif key in MOVES:
            current_move, last_key_time = MOVES[key], now

        if now - last_key_time > HOLD_TIME:
            current_move = (0, 0, 0, 0)          # no key -> hover in place
        if flying:
            tello.send_rc_control(*current_move)  # also keeps the drone awake

    print("Final counts:", {n: count_photos(n) for n in MEMBERS + ["background"]})

finally:
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
