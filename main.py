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

        elif command == "l" and tello.is_flying:
            tello.land()
            needs_landing = False

        # Move 30 cm per command. These calls block until the move finishes.
        elif command == "w" and tello.is_flying:
            tello.move_forward(30)
        elif command == "s" and tello.is_flying:
            tello.move_back(30)
        elif command == "a" and tello.is_flying:
            tello.move_left(30)
        elif command == "d" and tello.is_flying:
            tello.move_right(30)
        elif command == "r" and tello.is_flying:
            tello.move_up(30)
        elif command == "f" and tello.is_flying:
            tello.move_down(30)

        # Turn 30 degrees per command. These also block until the turn finishes.
        elif command == "q" and tello.is_flying:
            tello.rotate_counter_clockwise(30)
        elif command == "e" and tello.is_flying:
            tello.rotate_clockwise(30)

        else:
            print("Unknown command, or not allowed right now (take off first).")

        last_command_time = time.monotonic()

except Exception as e:
    print("Error:", e)

finally:
    if needs_landing:
        tello.land()
    tello.end()
