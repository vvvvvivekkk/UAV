# UAV — Tello Keyboard Control + YOLO26

## Quick setup (installs everything: djitellopy + YOLO)

- **Windows:** double-click `setup_windows.bat`, then double-click `run_demo.bat`.
- **macOS / Linux:** `bash setup_mac_linux.sh`, then `.venv/bin/python demo.py`.

Everything installs into a `.venv` folder inside this project, so nothing else
on the computer changes. Delete `.venv` to remove it all.
The first run of `demo.py` downloads `yolo26n.pt` and saves an image with boxes
to `runs/detect/predict/`.

| File | What it does |
|------|--------------|
| `main.py` | Fly the Tello with the keyboard |
| `demo.py` | First YOLO26 detection on a sample photo |
| `train.py` | Train YOLO26 on your own photos in `my-dataset/` |
| `predict_mine.py` | Run your trained model |
| `my-dataset/` | Put photos in `images/train` + `images/val`, labels in `labels/train` + `labels/val` |

---

Connect to a DJI Tello drone and fly it with single-key keyboard commands.
Each command moves **30 cm** or turns **30 degrees**, and the program waits
for the drone to finish before accepting the next key.

## Controls

| Key | Action        | Key | Action         |
|-----|---------------|-----|----------------|
| T   | take off      | L   | land           |
| W   | forward 30 cm | S   | back 30 cm     |
| A   | left 30 cm    | D   | right 30 cm    |
| R   | up 30 cm      | F   | down 30 cm     |
| Q   | turn left 30° | E   | turn right 30° |
| X   | exit          |     |                |

Press **Enter** after each key.

## Setup

### 1. Create a virtual environment

**Windows**
```
py -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Set up the .env file
Copy the example and fill in anything your class needs:
```
cp .env.example .env
```
(On Windows: `copy .env.example .env`)

The Tello creates its own Wi-Fi network, so no IP is needed by default.
`.env` is git-ignored so your settings stay local.

## Connect to the drone's Wi-Fi

The Tello has no internet — it is its own Wi-Fi access point, and your computer
must join that network before `main.py` can talk to it.

1. Press the **power button** on the side of the Tello once. The status light
   blinks yellow when it is ready.
2. On your computer, open **Wi-Fi settings**:
   - **Windows:** click the Wi-Fi icon in the taskbar (bottom-right).
   - **macOS:** click the Wi-Fi icon in the menu bar (top-right).
3. Select the network named **`TELLO-XXXXXX`** (the X's are your drone's ID).
   It is **open — no password**.
4. Wait until it shows **Connected** (you will have no internet while on it —
   that is normal).
5. Only one computer can control the drone at a time, so make sure no one else
   is connected to the same Tello.

## Run

1. Make sure you are connected to the Tello's Wi-Fi (see above).
2. Run:
```
python main.py
```

Land the drone before closing. The Tello may auto-land after ~15s of no commands.
