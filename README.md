# UAV — Tello Keyboard Control

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

## Run

1. Turn on the Tello and connect your computer to its **Wi-Fi** (TELLO-XXXXXX).
2. Run:
```
python main.py
```

Land the drone before closing. The Tello may auto-land after ~15s of no commands.
