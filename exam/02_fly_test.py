r"""
STEP 2 - Keyboard flying with the live camera.

Click on the video window first, then use these keys (no Enter needed):
  t = take off            l = land
  w / s = forward / back  a / d = left / right
  r / f = up / down       q / e = turn left / turn right
  SPACE = HOVER (stop moving)
  x = EMERGENCY motors off  (ONLY if it is about to hit a person!)
  ESC = land and quit

When you let go of the keys, the drone HOVERS in place.

Run (from the UAV folder):
  .venv\Scripts\python.exe exam\02_fly_test.py
"""
import time

import cv2
from djitellopy import Tello

# ---------------- SETTINGS ----------------
SPEED = 30            # movement speed (10-100). 30 is calm and safe indoors.
MIN_BATTERY = 20      # refuse to take off below this battery %
HOLD_TIME = 0.15      # seconds a key press keeps moving the drone
# ------------------------------------------

# Which key moves which way: (left/right, forward/back, up/down, turn)
MOVES = {
    ord("w"): (0, SPEED, 0, 0),   ord("s"): (0, -SPEED, 0, 0),
    ord("a"): (-SPEED, 0, 0, 0),  ord("d"): (SPEED, 0, 0, 0),
    ord("r"): (0, 0, SPEED, 0),   ord("f"): (0, 0, -SPEED, 0),
    ord("q"): (0, 0, 0, -SPEED),  ord("e"): (0, 0, 0, SPEED),
}

tello = Tello()
flying = False

try:
    # 1) Connect and start the camera.
    tello.connect()
    print("Battery:", tello.get_battery(), "%")
    tello.streamon()
    frame_reader = tello.get_frame_read()
    time.sleep(2)

    current_move = (0, 0, 0, 0)   # what the drone is doing right now
    last_key_time = 0
    status_text = "LANDED - press t to take off"

    # 2) Main loop: show video, read a key, send a movement command.
    while True:
        frame = frame_reader.frame
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB -> BGR for OpenCV
            battery = tello.get_battery()
            height = tello.get_height()
            cv2.putText(frame, f"Battery {battery}%  Height {height}cm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, status_text, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Tello fly test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("t") and not flying:
            if tello.get_battery() < MIN_BATTERY:
                status_text = f"Battery too low (< {MIN_BATTERY}%) - no takeoff"
            else:
                tello.takeoff()
                flying = True
                status_text = "FLYING"
        elif key == ord("l") and flying:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
            flying = False
            status_text = "LANDED - press t to take off"
        elif key == ord("x"):
            tello.emergency()      # motors stop instantly - drone falls!
            flying = False
            status_text = "EMERGENCY STOP"
        elif key == 27:            # ESC
            break
        elif key == ord(" "):
            current_move = (0, 0, 0, 0)
            status_text = "FLYING - HOVER" if flying else status_text
        elif key in MOVES:
            current_move = MOVES[key]
            last_key_time = time.time()

        # If no movement key for a short moment, stop moving (hover).
        if time.time() - last_key_time > HOLD_TIME:
            current_move = (0, 0, 0, 0)

        # Send the movement every loop. This also keeps the drone awake
        # (a Tello lands by itself after 15 s with no commands).
        if flying:
            tello.send_rc_control(*current_move)

finally:
    # 3) Safety: whatever happens, try to land and close everything.
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
