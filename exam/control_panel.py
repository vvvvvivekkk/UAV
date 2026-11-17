r"""
CONTROL PANEL - one window with live video, buttons, a command box and a log.
Use this instead of pressing keys in a hidden video window.

  Left : live Tello camera
  Right: buttons (click them, or hold the keyboard keys while this window is active)
         - Flying: Take off, Land, Hover, Emergency, and HOLD a move button to move
         - Photos: choose vivek / praney / empty room, Auto-capture, One photo
  Bottom: type a Tello command (e.g. "battery?", "up 30", "cw 90") + Send
          log of everything, with a Clear button

Keyboard (hold to move): w/s forward/back, a/d left/right, r/f up/down,
q/e turn, SPACE hover, t take off, l land, 1/2/0 choose person, g auto, c photo

Run (from the UAV folder):
  .venv2\Scripts\python.exe exam\control_panel.py
"""
import threading
import time
import tkinter as tk
from pathlib import Path

import cv2
from djitellopy import Tello
from PIL import Image, ImageTk

# ---------------- SETTINGS ----------------
MEMBERS = ["vivek", "praney"]
SAVE_DIR = Path(__file__).parent / "dataset_raw"
SPEED = 25            # flying speed (slow and safe)
MIN_BATTERY = 20
AUTO_DELAY = 5        # seconds to walk to your spot after Auto-capture
AUTO_SECONDS = 25     # how long auto-capture runs
AUTO_GAP = 0.4        # seconds between auto photos
VIDEO_W = 720         # width of the video in the window
# ------------------------------------------

MOVES = {"w": (0, SPEED, 0, 0), "s": (0, -SPEED, 0, 0), "a": (-SPEED, 0, 0, 0), "d": (SPEED, 0, 0, 0),
         "r": (0, 0, SPEED, 0), "f": (0, 0, -SPEED, 0), "q": (0, 0, 0, -SPEED), "e": (0, 0, 0, SPEED)}


class Panel:
    def __init__(self, root):
        self.root = root
        self.tello = Tello()
        self.flying = False
        self.move = (0, 0, 0, 0)
        self.person = MEMBERS[0]
        self.auto_start = None
        self.last_auto = 0
        self.frame = None              # latest clean BGR frame
        self.build_ui()
        self.log("Connecting to Tello... (laptop must be on TELLO-XXXXXX Wi-Fi)")
        threading.Thread(target=self.connect, daemon=True).start()

    # ---------- window layout ----------
    def build_ui(self):
        self.root.title("Tello Control Panel")
        self.root.attributes("-topmost", True)
        left = tk.Frame(self.root); left.grid(row=0, column=0, padx=6, pady=6)
        right = tk.Frame(self.root); right.grid(row=0, column=1, padx=6, pady=6, sticky="n")

        self.video = tk.Label(left, text="waiting for video...", width=90, height=30, bg="black", fg="white")
        self.video.pack()
        self.status = tk.Label(left, text="", font=("Arial", 12, "bold"))
        self.status.pack(fill="x")

        big = {"font": ("Arial", 12, "bold"), "width": 11, "height": 2}
        tk.Label(right, text="FLY", font=("Arial", 13, "bold")).grid(row=0, column=0, columnspan=3)
        tk.Button(right, text="Take off (t)", bg="#2e7d32", fg="white", command=self.takeoff, **big).grid(row=1, column=0)
        tk.Button(right, text="HOVER (space)", bg="#f9a825", command=self.hover, **big).grid(row=1, column=1)
        tk.Button(right, text="Land (l)", bg="#1565c0", fg="white", command=self.land, **big).grid(row=1, column=2)

        # Hold-to-move buttons
        layout = [("Up (r)", "r", 2, 0), ("Forward (w)", "w", 2, 1), ("Down (f)", "f", 2, 2),
                  ("Left (a)", "a", 3, 0), ("Back (s)", "s", 3, 1), ("Right (d)", "d", 3, 2),
                  ("Turn L (q)", "q", 4, 0), ("", None, 4, 1), ("Turn R (e)", "e", 4, 2)]
        for text, key, r, c in layout:
            if not key:
                continue
            b = tk.Button(right, text=text, **big)
            b.grid(row=r, column=c)
            b.bind("<ButtonPress-1>", lambda e, k=key: self.start_move(k))
            b.bind("<ButtonRelease-1>", lambda e: self.hover())
        tk.Button(right, text="EMERGENCY\nmotors off", bg="#c62828", fg="white", command=self.emergency,
                  **big).grid(row=4, column=1)

        tk.Label(right, text="PHOTOS", font=("Arial", 13, "bold")).grid(row=5, column=0, columnspan=3, pady=(12, 0))
        self.person_var = tk.StringVar(value=self.person)
        for i, name in enumerate(MEMBERS + ["background"]):
            tk.Radiobutton(right, text=f"{name} ({i + 1 if name != 'background' else 0})", variable=self.person_var,
                           value=name, font=("Arial", 12), indicatoron=False, width=11, height=2,
                           command=lambda: self.set_person(self.person_var.get())).grid(row=6, column=i)
        tk.Button(right, text="AUTO-CAPTURE (g)", bg="#6a1b9a", fg="white", command=self.toggle_auto,
                  **big).grid(row=7, column=0, columnspan=2, sticky="we")
        tk.Button(right, text="One photo (c)", command=self.one_photo, **big).grid(row=7, column=2)
        self.counts = tk.Label(right, text="", font=("Arial", 11), justify="left")
        self.counts.grid(row=8, column=0, columnspan=3, sticky="w")

        tk.Label(right, text="TYPE A TELLO COMMAND", font=("Arial", 13, "bold")).grid(row=9, column=0, columnspan=3, pady=(12, 0))
        self.cmd = tk.Entry(right, font=("Consolas", 12), width=24)
        self.cmd.grid(row=10, column=0, columnspan=2, sticky="we")
        self.cmd.bind("<Return>", lambda e: self.send_typed())
        tk.Button(right, text="Send", command=self.send_typed, width=11).grid(row=10, column=2)
        tk.Label(right, text='examples: battery?   temp?   up 30   cw 90   flip f', fg="gray").grid(row=11, column=0, columnspan=3)

        logf = tk.Frame(self.root); logf.grid(row=1, column=0, columnspan=2, sticky="we", padx=6, pady=(0, 6))
        self.logbox = tk.Text(logf, height=8, font=("Consolas", 10), bg="#111", fg="#0f0")
        self.logbox.pack(side="left", fill="x", expand=True)
        tk.Button(logf, text="Clear log", command=lambda: self.logbox.delete("1.0", "end")).pack(side="right", fill="y")

        # Keyboard: hold to move, release to hover
        self.root.bind("<KeyPress>", self.on_key)
        self.root.bind("<KeyRelease>", self.on_key_up)
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

    def count(self, name):
        folder = SAVE_DIR / name
        return len(list(folder.glob("*.jpg"))) if folder.exists() else 0

    def save(self, name):
        if self.frame is None:
            return None
        folder = SAVE_DIR / name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}_{time.strftime('%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"
        cv2.imwrite(str(path), self.frame)   # clean frame, no text on it
        return path

    # ---------- drone actions ----------
    def connect(self):
        try:
            self.tello.connect()
            self.log(f"Connected. Battery {self.tello.get_battery()}%")
            self.tello.streamon()
            self.reader = self.tello.get_frame_read()
            time.sleep(1.5)
            self.root.after(0, self.update_video)
            self.root.after(0, self.control_loop)
        except Exception as ex:
            self.log(f"CONNECT FAILED: {ex}  -> check Wi-Fi is TELLO-XXXXXX, restart drone, run again")

    def takeoff(self):
        if self.flying:
            return
        if self.tello.get_battery() < MIN_BATTERY:
            self.log("Battery too low - no takeoff")
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

    def set_person(self, name):
        self.person = name
        self.person_var.set(name)
        self.log(f"Selected: {name}")

    def toggle_auto(self):
        if self.auto_start is None:
            self.auto_start = time.time()
            self.log(f"Auto-capture for {self.person}: go to your spot, starting in {AUTO_DELAY}s")
        else:
            self.auto_start = None
            self.log("Auto-capture stopped")

    def one_photo(self):
        p = self.save(self.person)
        self.log(f"Saved {p.name}" if p else "No video yet")

    def send_typed(self):
        text = self.cmd.get().strip()
        if not text:
            return
        self.cmd.delete(0, "end")
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
        elif k in ("1", "2"):
            self.set_person(MEMBERS[int(k) - 1])
        elif k == "0":
            self.set_person("background")
        elif k == "g":
            self.toggle_auto()
        elif k == "c":
            self.one_photo()

    def on_key_up(self, e):
        if e.keysym.lower() in MOVES:
            self.hover()

    # ---------- loops ----------
    def control_loop(self):
        """Every 50 ms: send movement (keeps the drone awake) and run auto-capture."""
        if self.flying:
            try:
                self.tello.send_rc_control(*self.move)
            except Exception:
                pass
        if self.auto_start is not None:
            t = time.time() - self.auto_start
            if AUTO_DELAY <= t < AUTO_DELAY + AUTO_SECONDS and time.time() - self.last_auto >= AUTO_GAP:
                self.save(self.person)
                self.last_auto = time.time()
            elif t >= AUTO_DELAY + AUTO_SECONDS:
                self.auto_start = None
                self.log(f"Auto-capture done: {self.person} now has {self.count(self.person)} photos")
        self.root.after(50, self.control_loop)

    def update_video(self):
        f = self.reader.frame
        if f is not None:
            self.frame = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)   # clean copy for saving
            h = int(f.shape[0] * VIDEO_W / f.shape[1])
            img = ImageTk.PhotoImage(Image.fromarray(f).resize((VIDEO_W, h)))   # f is RGB already
            self.video.configure(image=img, text="", width=VIDEO_W, height=h)
            self.video.image = img
        auto = ""
        if self.auto_start is not None:
            t = time.time() - self.auto_start
            auto = f"  |  GO TO SPOT {AUTO_DELAY - t:.0f}s" if t < AUTO_DELAY else f"  |  CAPTURING {AUTO_DELAY + AUTO_SECONDS - t:.0f}s"
        try:
            self.status.configure(text=f"Battery {self.tello.get_battery()}%   Temp {self.tello.get_temperature()}C   "
                                       f"{'FLYING' if self.flying else 'LANDED'}   Person: {self.person}{auto}",
                                  fg="red" if auto else "black")
        except Exception:
            pass
        self.counts.configure(text="   ".join(f"{n}: {self.count(n)}" for n in MEMBERS + ["background"]))
        self.root.after(40, self.update_video)

    def quit(self):
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
    Panel(root)
    root.mainloop()
