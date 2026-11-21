r"""
DRONE CHECK - test everything before flying: Wi-Fi, connection, commands,
live video, and live recognition. Prints PASS / WARN / FAIL for each step
and saves a report + snapshots in exam\check_results\.

The drone does NOT fly (motors stay off) unless you add --fly.

Run (from the UAV folder, laptop on the TELLO-XXXXXX Wi-Fi):
  .venv2\Scripts\python.exe exam\drone_check.py          # no flying
  .venv2\Scripts\python.exe exam\drone_check.py --fly    # + short test flight

During the video tests a window shows the live camera: stand 1-2 m in front
of the drone and look at it, so the recognition test can see a face.

--fly test (needs space, ~2 m clear around + above):
  take off -> hover 3 s -> up 30 cm -> turn right 90 -> turn left 90 -> land.
  Ctrl+C at any time = land immediately.
"""
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from djitellopy import Tello

# ---------------- SETTINGS ----------------
HERE = Path(__file__).parent
OUT = HERE / "check_results"
VIDEO_SECONDS = 10        # how long to measure the video
RECOG_SECONDS = 10        # how long to test live recognition
MIN_FLY_BATTERY = 30      # % needed for the --fly test
MAX_TEMP = 80             # C, above this: let it cool down
# ------------------------------------------

results = []              # (step, status, details)


def report(step, status, details=""):
    results.append((step, status, details))
    mark = {"PASS": "[ PASS ]", "WARN": "[ WARN ]", "FAIL": "[ FAIL ]", "INFO": "[ INFO ]"}[status]
    print(f"{mark} {step}" + (f": {details}" if details else ""))


def try_get(fn):
    try:
        return fn()
    except Exception as ex:
        return f"(error: {ex})"


def check_wifi():
    """Windows only: which Wi-Fi is the laptop on? Must be TELLO-XXXXXX."""
    if sys.platform != "win32":
        report("1. Laptop Wi-Fi", "INFO", "not Windows - make sure you are on TELLO-XXXXXX")
        return
    try:
        out = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True, timeout=5).stdout
        ssid = next((l.split(":", 1)[1].strip() for l in out.splitlines()
                     if l.strip().startswith("SSID") and "BSSID" not in l), None)
    except Exception:
        ssid = None
    if ssid and ssid.upper().startswith("TELLO"):
        report("1. Laptop Wi-Fi", "PASS", ssid)
    elif ssid:
        report("1. Laptop Wi-Fi", "FAIL", f"connected to '{ssid}', not the drone. Join TELLO-XXXXXX")
    else:
        report("1. Laptop Wi-Fi", "WARN", "could not read the Wi-Fi name - check it is TELLO-XXXXXX")


def show(frame, text, window="Drone check (q = skip)"):
    view = frame.copy()
    cv2.putText(view, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow(window, view)
    cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    return (cv2.waitKey(1) & 0xFF) == ord("q")


def main():
    fly = "--fly" in sys.argv
    OUT.mkdir(exist_ok=True)
    print("=" * 60 + "\nDRONE CHECK  " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + "=" * 60)

    check_wifi()

    # 2) Connect --------------------------------------------------------
    tello = Tello()
    try:
        t = time.time()
        tello.connect()
        report("2. Connect", "PASS", f"answered in {time.time() - t:.1f} s")
    except Exception as ex:
        report("2. Connect", "FAIL", f"{ex}  -> drone on? laptop on TELLO Wi-Fi? only ONE laptop connected?")
        return finish(None)

    try:
        # 3) Status ----------------------------------------------------
        battery = tello.get_battery()
        temp = tello.get_temperature()
        report("3. Battery", "PASS" if battery >= MIN_FLY_BATTERY else "WARN",
               f"{battery}%" + ("" if battery >= MIN_FLY_BATTERY else " - charge before the exam"))
        report("3. Temperature", "PASS" if temp < MAX_TEMP else "WARN",
               f"{temp} C" + ("" if temp < MAX_TEMP else " - too hot, turn off 10 min"))
        report("3. Height / flight time", "INFO", f"{tello.get_height()} cm, {tello.get_flight_time()} s")

        # 4) Read commands (ask the drone questions) ---------------------
        for cmd in ["battery?", "sdk?", "sn?", "wifi?", "time?"]:
            ans = try_get(lambda: tello.send_read_command(cmd))
            ok = not str(ans).startswith("(error")
            report(f"4. Command '{cmd}'", "PASS" if ok else "WARN", str(ans))

        # 5) Control command that does NOT fly (set speed) ----------------
        ok = try_get(lambda: tello.send_control_command("speed 30"))
        report("5. Control command 'speed 30'", "PASS" if ok is True else "FAIL", str(ok))

        # 6) Video ------------------------------------------------------
        tello.streamon()
        reader = tello.get_frame_read()
        t = time.time()
        while reader.frame is None and time.time() - t < 10:
            time.sleep(0.1)
        if reader.frame is None:
            report("6. Video start", "FAIL", "no picture in 10 s -> allow python.exe in Windows Firewall")
        else:
            report("6. Video start", "PASS", f"first picture after {time.time() - t:.1f} s")
            frames, last, t = 0, None, time.time()
            bright = []
            while time.time() - t < VIDEO_SECONDS:
                f = reader.frame
                if f is not None and f is not last:          # a NEW picture arrived
                    frames, last = frames + 1, f
                    bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                    bright.append(float(bgr.mean()))
                    if show(bgr, f"VIDEO TEST {VIDEO_SECONDS - (time.time() - t):.0f}s  battery {tello.get_battery()}%"):
                        break
                time.sleep(0.005)
            fps = frames / max(0.1, time.time() - t)
            h, w = last.shape[:2]
            report("6. Video speed", "PASS" if fps >= 15 else "WARN",
                   f"{fps:.1f} pictures/s at {w}x{h}" + ("" if fps >= 15 else " - weak Wi-Fi? move closer, fewer phones nearby"))
            dark = np.mean(bright) < 15
            report("6. Picture not black", "WARN" if dark else "PASS",
                   f"brightness {np.mean(bright):.0f}/255" + (" - lens covered or room dark?" if dark else ""))
            snap = OUT / "video_snapshot.jpg"
            cv2.imwrite(str(snap), cv2.cvtColor(last, cv2.COLOR_RGB2BGR))
            report("6. Snapshot saved", "PASS", str(snap))

            # 7) Live recognition --------------------------------------
            print("\n>>> Recognition test: stand 1-2 m in front of the drone and LOOK at it <<<")
            try:
                sys.path.insert(0, str(HERE))
                import identify as ident
                ident.identify(np.zeros((720, 960, 3), np.uint8))       # warm up
                ident.identity_model.predict(np.zeros((160, 120, 3), np.uint8), verbose=False)
                n, seen, best, t = 0, {}, None, time.time()
                while time.time() - t < RECOG_SECONDS:
                    bgr = cv2.cvtColor(reader.frame, cv2.COLOR_RGB2BGR)
                    res = ident.identify(bgr)
                    n += 1
                    for label, conf, _ in res:
                        seen[label] = max(seen.get(label, 0), conf or 0)
                        if conf is not None and (best is None or conf > best[0]):
                            best = (conf, bgr.copy(), res)
                    if show(ident.draw(bgr.copy(), res), f"RECOGNITION TEST {RECOG_SECONDS - (time.time() - t):.0f}s"):
                        break
                rate = n / max(0.1, time.time() - t)
                report("7. Recognition speed", "PASS" if rate >= 3 else "WARN", f"{rate:.1f} frames/s")
                members = {k: v for k, v in seen.items() if k in ident.MEMBERS}
                if members:
                    txt = ", ".join(f"{k} best {v:.2f}" for k, v in members.items())
                    report("7. Member recognised", "PASS" if max(members.values()) >= 0.80 else "WARN", txt)
                    path = OUT / "recognition_snapshot.jpg"
                    cv2.imwrite(str(path), ident.draw(best[1], best[2]))
                    report("7. Recognition snapshot", "PASS", str(path))
                else:
                    report("7. Member recognised", "WARN",
                           "no vivek/praney seen" + (" (only 'other person')" if seen else " (nobody seen)")
                           + " - stand in front, face the drone, good light")
            except Exception as ex:
                report("7. Recognition", "FAIL", f"{ex} -> run exam\\check_setup.py")

        # 8) Optional short flight ----------------------------------------
        if fly:
            battery, temp = tello.get_battery(), tello.get_temperature()
            if battery < MIN_FLY_BATTERY or temp >= MAX_TEMP:
                report("8. Flight test", "FAIL", f"skipped: battery {battery}%, temp {temp} C")
            elif input("\nFLIGHT TEST: 2 m clear space? Type YES to take off: ").strip().upper() != "YES":
                report("8. Flight test", "INFO", "skipped by you")
            else:
                steps = [("take off", tello.takeoff), ("hover 3 s", lambda: time.sleep(3)),
                         ("up 30 cm", lambda: tello.move_up(30)), ("turn right 90", lambda: tello.rotate_clockwise(90)),
                         ("turn left 90", lambda: tello.rotate_counter_clockwise(90)), ("land", tello.land)]
                try:
                    for name, fn in steps:
                        print(f"  -> {name}")
                        fn()
                        report(f"8. Fly: {name}", "PASS")
                except KeyboardInterrupt:
                    report("8. Flight test", "WARN", "stopped with Ctrl+C - landing")
                    tello.land()
                except Exception as ex:
                    report("8. Flight test", "FAIL", f"{ex} - landing")
                    try_get(tello.land)
        else:
            report("8. Flight test", "INFO", "not run (add --fly to test take off / move / land)")
    except KeyboardInterrupt:
        report("Stopped", "WARN", "Ctrl+C")
        try_get(tello.land)
    finally:
        try_get(tello.streamoff)
        try_get(tello.end)
        cv2.destroyAllWindows()
    finish(tello)


def finish(_):
    fails = [r for r in results if r[1] == "FAIL"]
    warns = [r for r in results if r[1] == "WARN"]
    verdict = ("READY TO FLY" if not fails and not warns else
               "OK, BUT CHECK THE WARNINGS" if not fails else "NOT READY - fix the FAIL lines")
    print("=" * 60 + f"\nRESULT: {verdict}   ({len(fails)} fail, {len(warns)} warning)\n" + "=" * 60)
    lines = [f"{s:5} {step}: {d}" for step, s, d in results]
    (OUT / "report.txt").write_text(f"DRONE CHECK {time.strftime('%Y-%m-%d %H:%M:%S')}\n" + "\n".join(lines)
                                    + f"\nRESULT: {verdict}\n", encoding="utf-8")
    print(f"Report saved: {OUT / 'report.txt'}")


if __name__ == "__main__":
    main()
