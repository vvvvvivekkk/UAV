r"""
STEP 1 - Connect test (the drone does NOT fly).

What it does:
  1. Connects to the Tello over its Wi-Fi.
  2. Prints battery %, temperature and Wi-Fi signal.
  3. Shows the live camera for 20 seconds (press q to quit early).

Run (from the UAV folder):
  .venv2\Scripts\python.exe exam\01_connect_test.py
"""
import time

import cv2
from djitellopy import Tello

# ---------------- SETTINGS ----------------
SHOW_SECONDS = 20   # how long to show the video
# ------------------------------------------

tello = Tello()

try:
    # 1) Connect. If this fails, your laptop is not on the TELLO-XXXXXX Wi-Fi.
    print("Connecting to Tello...")
    tello.connect()
    print("Connected!")

    # 2) Read the drone's status.
    print("Battery     :", tello.get_battery(), "%")
    print("Temperature :", tello.get_temperature(), "C")
    try:
        print("Wi-Fi SNR   :", tello.query_wifi_signal_noise_ratio())
    except Exception:
        print("Wi-Fi SNR   : (not available on this firmware)")

    # 3) Turn on the camera stream and show it.
    tello.streamon()
    frame_reader = tello.get_frame_read()
    time.sleep(2)  # give the video a moment to start

    start = time.time()
    while time.time() - start < SHOW_SECONDS:
        frame = frame_reader.frame
        if frame is None:
            continue
        # djitellopy gives RGB colours; OpenCV wants BGR, so convert.
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(frame, f"Battery: {tello.get_battery()}%  (q = quit)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Tello connect test", frame)
        cv2.setWindowProperty("Tello connect test", cv2.WND_PROP_TOPMOST, 1)  # keep window in front so it gets the keys
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("Video test finished OK.")

finally:
    # Always clean up, even if something went wrong above.
    try:
        tello.streamoff()
    except Exception:
        pass
    tello.end()
    cv2.destroyAllWindows()
