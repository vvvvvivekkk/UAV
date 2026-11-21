r"""
FLIGHT CONSOLE - one window to fly the Tello AND recognise the person live.
This is the program for the exam.

  Left : live Tello camera with boxes:  "vivek 0.97" / "praney 0.95" / "other person"
         big status line: battery, height, temperature, FLYING/LANDED, SAVED OK
  Right: buttons (click, or use the keyboard while this window is active)
         - FLY: Take off, Hover, Land, EMERGENCY, hold a move button to move
         - PHOTO: Save now, New flight (ready to save again)
         - type a Tello command (e.g. "battery?", "up 30", "cw 90") + Send
  Bottom: log of everything, Clear log

AUTO-SAVE (exam rule: photo with name + box + RAW confidence must be saved):
  when the SAME member is seen STABLE_FRAMES times in a row with confidence
  >= SAVE_MIN_CONF (0.80 -> recognition score >= 100), it keeps watching for
  BEST_WINDOW seconds, saves the frame with the HIGHEST raw confidence to
  exam\exam_photos\, re-opens it to check, and shows "SAVED OK".
  Can't reach 0.80? Press p (Save now): it saves the best member frame of the
  last few seconds (still the raw confidence, never changed).

Keyboard (click the window first; hold to move, let go = hover):
  t take off | l land | SPACE hover | x EMERGENCY (motors off)
  w/s forward/back | a/d left/right | r/f up/down | q/e turn
  p save photo now | n new flight (allow a new auto-save)

Run (from the UAV folder):
  .venv2\Scripts\python.exe exam\flight_console.py
"""
import threading
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
from djitellopy import Tello
from PIL import Image, ImageTk

import identify as ident   # our face recogniser (same folder)

# ---------------- SETTINGS ----------------
HERE = Path(__file__).parent
PHOTO_DIR = HERE / "exam_photos"
SPEED = 30            # flying speed (10-100)
MIN_BATTERY = 20      # no takeoff below this %
SAVE_MIN_CONF = 0.80  # auto-save only at this raw confidence or higher (score >= 100)
STABLE_FRAMES = 3     # same member this many recognitions in a row before saving
BEST_WINDOW = 1.0     # seconds to look for the clearest frame, then save
MANUAL_LOOKBACK = 3.0 # "Save now" picks the best member frame of the last N seconds
VIDEO_W = 800         # width of the video in the window
# ------------------------------------------

MOVES = {"w": (0, SPEED, 0, 0), "s": (0, -SPEED, 0, 0), "a": (-SPEED, 0, 0, 0), "d": (SPEED, 0, 0, 0),
         "r": (0, 0, SPEED, 0), "f": (0, 0, -SPEED, 0), "q": (0, 0, 0, -SPEED), "e": (0, 0, 0, SPEED)}


def best_member(results):
    """Team member with the highest RAW confidence as (name, conf, box), or None."""
    members = [r for r in results if r[1] is not None]   # "other person" has no conf
    return max(members, key=lambda r: r[1]) if members else None


class Console:
    def __init__(self, root):
        self.root = root
        self.tello = Tello()
        self.flying = False
        self.running = True
        self.move = (0, 0, 0, 0)
        self.frame = None                 # latest clean BGR frame from the drone
        self.results = []                 # latest recognition results (for drawing)
        self.rec_fps = 0.0
        self.streak_name, self.streak = None, 0
        self.window_start, self.window_best = None, None
        self.recent = []                  # (time, frame, results, det) for "Save now"
        self.saved_text = ""
        self.lock = threading.Lock()
        self.build_ui()
        self.log("Loading models...")
        threading.Thread(target=self.start, daemon=True).start()

    # ---------- window layout ----------
    def build_ui(self):
        self.root.title("Tello Flight Console - live recognition")
        self.root.attributes("-topmost", True)
        left = tk.Frame(self.root); left.grid(row=0, column=0, padx=6, pady=6)
        right = tk.Frame(self.root); right.grid(row=0, column=1, padx=6, pady=6, sticky="n")

        self.video = tk.Label(left, text="waiting for video...", width=100, height=34, bg="black", fg="white")
        self.video.pack()
        self.status = tk.Label(left, text="", font=("Arial", 13, "bold"))
        self.status.pack(fill="x")
        self.seen = tk.Label(left, text="", font=("Arial", 13, "bold"))
        self.seen.pack(fill="x")
        self.saved = tk.Label(left, text="", font=("Arial", 15, "bold"), fg="#2e7d32")
        self.saved.pack(fill="x")

        big = {"font": ("Arial", 12, "bold"), "width": 11, "height": 2}
        tk.Label(right, text="FLY", font=("Arial", 13, "bold")).grid(row=0, column=0, columnspan=3)
        tk.Button(right, text="Take off (t)", bg="#2e7d32", fg="white", command=self.takeoff, **big).grid(row=1, column=0)
        tk.Button(right, text="HOVER (space)", bg="#f9a825", command=self.hover, **big).grid(row=1, column=1)
        tk.Button(right, text="Land (l)", bg="#1565c0", fg="white", command=self.land, **big).grid(row=1, column=2)
        layout = [("Up (r)", "r", 2, 0), ("Forward (w)", "w", 2, 1), ("Down (f)", "f", 2, 2),
                  ("Left (a)", "a", 3, 0), ("Back (s)", "s", 3, 1), ("Right (d)", "d", 3, 2),
                  ("Turn L (q)", "q", 4, 0), ("Turn R (e)", "e", 4, 2)]
        for text, key, r, c in layout:
            b = tk.Button(right, text=text, **big)
            b.grid(row=r, column=c)
            b.bind("<ButtonPress-1>", lambda e, k=key: self.start_move(k))
            b.bind("<ButtonRelease-1>", lambda e: self.hover())
        tk.Button(right, text="EMERGENCY (x)\nmotors off", bg="#c62828", fg="white", command=self.emergency,
                  **big).grid(row=4, column=1)

        tk.Label(right, text="PHOTO", font=("Arial", 13, "bold")).grid(row=5, column=0, columnspan=3, pady=(12, 0))
        tk.Button(right, text="Save now (p)", bg="#6a1b9a", fg="white", command=self.save_now,
                  **big).grid(row=6, column=0, columnspan=2, sticky="we")
        tk.Button(right, text="New flight (n)", command=self.new_flight, **big).grid(row=6, column=2)
        tk.Label(right, text=f"Auto-save at confidence >= {SAVE_MIN_CONF:.2f}\n(recognition score >= {SAVE_MIN_CONF * 100 + 20:.0f})",
                 fg="gray").grid(row=7, column=0, columnspan=3)

        tk.Label(right, text="TYPE A TELLO COMMAND", font=("Arial", 13, "bold")).grid(row=8, column=0, columnspan=3, pady=(12, 0))
        self.cmd = tk.Entry(right, font=("Consolas", 12), width=24)
        self.cmd.grid(row=9, column=0, columnspan=2, sticky="we")
        self.cmd.bind("<Return>", lambda e: self.send_typed())
        tk.Button(right, text="Send", command=self.send_typed, width=11).grid(row=9, column=2)
        tk.Label(right, text="examples: battery?  temp?  up 30  cw 90", fg="gray").grid(row=10, column=0, columnspan=3)
        tk.Label(right, text="Click the video to use the keyboard again", fg="gray").grid(row=11, column=0, columnspan=3)

        logf = tk.Frame(self.root); logf.grid(row=1, column=0, columnspan=2, sticky="we", padx=6, pady=(0, 6))
        self.logbox = tk.Text(logf, height=7, font=("Consolas", 10), bg="#111", fg="#0f0")
        self.logbox.pack(side="left", fill="x", expand=True)
        tk.Button(logf, text="Clear log", command=lambda: self.logbox.delete("1.0", "end")).pack(side="right", fill="y")

        self.root.bind("<KeyPress>", self.on_key)
        self.root.bind("<KeyRelease>", self.on_key_up)
        self.video.bind("<Button-1>", lambda e: self.root.focus_set())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    # ---------- helpers ----------
    def log(self, msg):
        self.root.after(0, lambda: (self.logbox.insert("end", time.strftime("%H:%M:%S ") + msg + "\n"),
                                    self.logbox.see("end")))

    def run_bg(self, fn, label):
        """Run slow drone commands (takeoff/land) without freezing the window."""
        def work():
            try:
                fn()
                self.log(label + " OK")
            except Exception as ex:
                self.log(f"{label} FAILED: {ex}")
        threading.Thread(target=work, daemon=True).start()

    # ---------- start: models, then drone ----------
    def start(self):
        try:
            ident.identify(np.zeros((720, 960, 3), dtype=np.uint8))          # warm up (first run is slow)
            ident.identity_model.predict(np.zeros((160, 120, 3), dtype=np.uint8), verbose=False)
            self.log("Models ready. Connecting to Tello... (laptop must be on TELLO-XXXXXX Wi-Fi)")
            self.tello.connect()
            self.log(f"Connected. Battery {self.tello.get_battery()}%")
            self.tello.streamon()
            self.reader = self.tello.get_frame_read()
            time.sleep(1.5)
            threading.Thread(target=self.recognise_loop, daemon=True).start()
            self.root.after(0, self.update_video)
            self.root.after(0, self.control_loop)
        except Exception as ex:
            self.log(f"START FAILED: {ex}  -> check Wi-Fi is TELLO-XXXXXX, restart drone, run again")

    # ---------- drone actions ----------
    def takeoff(self):
        if self.flying:
            return
        try:
            battery = self.tello.get_battery()
        except Exception:
            battery = 0
        if battery < MIN_BATTERY:
            self.log(f"Battery {battery}% too low - no takeoff")
            return
        self.flying = True
        self.run_bg(self.tello.takeoff, "Take off")

    def land(self):
        self.move = (0, 0, 0, 0)
        if self.flying:
            self.flying = False
            self.run_bg(lambda: (self.tello.send_rc_control(0, 0, 0, 0), self.tello.land()), "Land")

    def hover(self):
        self.move = (0, 0, 0, 0)

    def emergency(self):
        self.flying = False
        self.move = (0, 0, 0, 0)
        self.run_bg(self.tello.emergency, "EMERGENCY motors off")

    def start_move(self, key):
        if self.flying:
            self.move = MOVES[key]

    def send_typed(self):
        text = self.cmd.get().strip()
        if not text:
            return
        self.cmd.delete(0, "end")
        self.root.focus_set()           # give the keyboard back to flying
        self.log(f"> {text}")
        def work():
            try:
                if text.endswith("?"):
                    self.log("< " + str(self.tello.send_read_command(text)))
                else:
                    self.log("< " + ("ok" if self.tello.send_control_command(text) else "error"))
            except Exception as ex:
                self.log(f"< FAILED: {ex}")
        threading.Thread(target=work, daemon=True).start()

    # ---------- photos ----------
    def save_photo(self, clean_frame, results, det):
        """Draw every label on a copy, save it, and check the file opens (exam rule)."""
        img = ident.draw(clean_frame.copy(), results)
        PHOTO_DIR.mkdir(exist_ok=True)
        name, conf, _ = det
        path = PHOTO_DIR / f"{name}_{conf:.3f}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        ok = cv2.imwrite(str(path), img) and cv2.imread(str(path)) is not None
        self.saved_text = (f"SAVED OK: {path.name}  (score {conf * 100 + 20:.0f})" if ok
                           else "SAVE FAILED - press p again")
        self.log(self.saved_text)
        return ok

    def save_now(self):
        with self.lock:
            now = time.time()
            recent = [r for r in self.recent if now - r[0] <= MANUAL_LOOKBACK]
        if not recent:
            self.log("Nothing recognised yet - get closer, face the person")
            return
        _, frame, results, det = max(recent, key=lambda r: r[3][1])
        self.save_photo(frame, results, det)

    def new_flight(self):
        self.saved_text = ""
        self.streak_name, self.streak = None, 0
        self.window_start, self.window_best = None, None
        self.log("New flight: auto-save is ready again")

    # ---------- keyboard ----------
    def on_key(self, e):
        if self.root.focus_get() == self.cmd:
            return                      # typing in the command box
        k = e.keysym.lower()
        if k in MOVES:
            self.start_move(k)
        elif k == "space":
            self.hover()
        elif k == "t":
            self.takeoff()
        elif k == "l":
            self.land()
        elif k == "x":
            self.emergency()
        elif k == "p":
            self.save_now()
        elif k == "n":
            self.new_flight()

    def on_key_up(self, e):
        if e.keysym.lower() in MOVES:
            self.hover()

    # ---------- loops ----------
    def recognise_loop(self):
        """Runs all the time in the background: recognise the newest frame, auto-save."""
        while self.running:
            f = self.reader.frame
            if f is None:
                time.sleep(0.01)
                continue
            clean = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)       # Tello gives RGB
            t0 = time.time()
            results = ident.identify(clean)
            self.rec_fps = 0.8 * self.rec_fps + 0.2 / max(1e-3, time.time() - t0)
            det = best_member(results)
            with self.lock:
                self.results = results
                if det:
                    self.recent.append((time.time(), clean, results, det))
                    self.recent = [r for r in self.recent if time.time() - r[0] <= MANUAL_LOOKBACK]
            self.auto_save(clean, results, det)

    def auto_save(self, clean, results, det):
        good = det is not None and det[1] >= SAVE_MIN_CONF
        if good and det[0] == self.streak_name:
            self.streak += 1
        else:
            self.streak_name, self.streak = (det[0] if good else None), (1 if good else 0)
            self.window_start, self.window_best = None, None
        if self.saved_text.startswith("SAVED OK") or self.streak < STABLE_FRAMES:
            return
        if self.window_start is None:
            self.window_start = time.time()
        if self.window_best is None or det[1] > self.window_best[2][1]:
            self.window_best = (clean, results, det)          # keep the raw best
        if time.time() - self.window_start >= BEST_WINDOW:
            self.save_photo(*self.window_best)

    def control_loop(self):
        """Every 50 ms send the movement (no key = hover; also keeps the drone awake)."""
        if self.flying:
            try:
                self.tello.send_rc_control(*self.move)
            except Exception:
                pass
        if self.running:
            self.root.after(50, self.control_loop)

    def update_video(self):
        f = self.reader.frame
        if f is not None:
            view = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            with self.lock:
                results = list(self.results)
            ident.draw(view, results)
            h = int(view.shape[0] * VIDEO_W / view.shape[1])
            rgb = cv2.cvtColor(cv2.resize(view, (VIDEO_W, h)), cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video.configure(image=img, text="", width=VIDEO_W, height=h)
            self.video.image = img
            det = best_member(results)
            if det:
                color = "#2e7d32" if det[1] >= SAVE_MIN_CONF else "#ef6c00"
                self.seen.configure(text=f"SEEN: {det[0]} {det[1]:.2f}  (score {det[1] * 100 + 20:.0f})"
                                         + ("" if det[1] >= SAVE_MIN_CONF else "  - get closer / hold steady"), fg=color)
            else:
                n_other = sum(1 for r in results if r[1] is None)
                self.seen.configure(text=f"SEEN: {n_other} other person(s)" if n_other else "SEEN: nobody", fg="gray")
        try:
            self.status.configure(text=f"Battery {self.tello.get_battery()}%   Height {self.tello.get_height()} cm   "
                                       f"Temp {self.tello.get_temperature()} C   "
                                       f"{'FLYING' if self.flying else 'LANDED'}   recognition {self.rec_fps:.1f}/s")
        except Exception:
            pass
        self.saved.configure(text=self.saved_text, fg="#2e7d32" if "OK" in self.saved_text else "#c62828")
        if self.running:
            self.root.after(40, self.update_video)

    def quit(self):
        self.running = False
        try:
            if self.flying:
                self.tello.send_rc_control(0, 0, 0, 0)
                self.tello.land()
            self.tello.streamoff()
        except Exception:
            pass
        try:
            self.tello.end()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    Console(root)
    root.mainloop()
