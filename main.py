import time
from djitellopy import Tello

tello = Tello(retry_count=1)
needs_landing = False

try:
    tello.connect()
    tello.set_speed(20)
    print("Battery:", tello.get_battery(), "%")
    print("T: take off, L: land, X: exit. Press Enter after each command.")
    print("W/S: forward/back, A/D: left/right, R/F: up/down. 30 cm each.")
    print("Q/E: turn left/right. 30 degrees each.")
    print("Land before leaving. Tello may auto-land after 15s of idle.")
    last_command_time = time.monotonic()

    while True:
        command = input().strip().lower()

        if command == "x":
            break

        elif command == "t" and not tello.is_flying:
            if tello.get_battery() >= 10:
                needs_landing = True
                tello.takeoff()
            else:
                print("Battery below 10%. Please charge first.")

        # more commands (L, W/S, A/D, R/F, Q/E) go here when you send the next photo

        last_command_time = time.monotonic()

except Exception as e:
    print("Error:", e)

finally:
    if needs_landing:
        tello.land()
    tello.end()
