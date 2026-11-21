# UAV Exam – Full Step-by-Step Guide (do it yourself)

Team: **vivek** (no glasses) and **praney** (glasses). Same clothes on photo day and exam day.
Always open a terminal **in the UAV folder** (`C:\Users\11020\Desktop\UAV`) and copy the commands exactly.

---

## Exam rules in short (from the teacher's slides, "preliminary rules")

- **Roles each flight:** pilot (back to the drone, watches ONLY the laptop live video), target (stands in the taped box), safety observer (watches the drone, can shout STOP, may not help fly). Rules say teams of 3 and everyone pilots once; **we are 2, so ask the teacher how we rotate**.
- **Route:** take off from the start square → U-shaped route (ㄇ) around the tables → target on the other side.
- **Timer:** starts at the teacher's "go", stops when the photo with **name + box + confidence** is **saved and opens**.
- **Score = 50% time + 50% recognition.** Recognition = **confidence% + 20** if the name is right (0.80 → 100, 1.00 → 120). Time: fastest team 100, slowest 80, others in between by rank. Best valid flight of 3 counts. Max 110.
- Confidence must be the model's **raw** output (never scaled/boosted/fixed) – our code does this.
- **0 points for that flight:** hit a person, crash, leave the area / skip the route, wrong name / no photo, not stopping when told.
- Image must come from the **Tello live stream**. Hand in code + models + photos right after; the teacher may ask you to explain the code (oral check).

---

## 0. One-time setup (needs normal internet, NOT the Tello Wi-Fi)

**Double-click `setup.bat`** in the UAV folder (or type `setup.bat` in a terminal there).
It creates ONE venv, `.venv2`, installs everything from `requirements.txt`
(YOLO26, Tello, OpenCV, Pillow, HEIC support), then runs the check and prints **ALL GOOD**.
Takes a few minutes the first time (downloads PyTorch). Do this on every new laptop (e.g. Praney's).

Check again any time (no drone needed):
```
.venv2\Scripts\python.exe exam\check_setup.py
```

**If you see "Access is denied" or "This app can't run on your PC":** the `python.exe` inside `.venv2` got damaged. Fix (10 s, keeps everything):
```
py -m venv --upgrade .venv2
```

---

## 1. Connect to the drone

1. Turn the Tello on, wait for the blinking light.
2. Connect the laptop Wi-Fi to **TELLO-XXXXXX** (no internet on this Wi-Fi – that's normal).
3. Test:
```
.venv2\Scripts\python.exe exam\01_connect_test.py
```
You should see `Connected!`, battery %, and a live video window for 20 s.

| Problem | Fix |
|---|---|
| `Did not receive a state packet` | Just run it again (normal on the first try) |
| `Aborting command 'command'` (no response) | Laptop left the Tello Wi-Fi, or drone turned off. Restart drone, reconnect Wi-Fi |
| No video | Allow `python.exe` in Windows Firewall (Private + Public) |
| Temp above ~80 °C | Turn drone off for 10 min. It shuts down at ~85–90 °C |

---

## 2. Fly + take photos (control panel)

```
.venv2\Scripts\python.exe exam\control_panel.py
```
**Click once on the panel window** so it gets your keys.

| Fly | Key / button |
|---|---|
| Take off / Land | **t** / **l** |
| Forward / back / left / right | hold **w / s / a / d** (let go = hover) |
| Up / down | hold **r / f** |
| Turn left / right | hold **q / e** |
| Hover | **space** |
| EMERGENCY (motors off – only if about to hit a person) | red button |

| Photos | Key / button |
|---|---|
| Choose person | **1** vivek · **2** praney · **0** background (empty room) |
| Auto-capture (5 s to walk to your spot, then 1 photo / 0.4 s for 25 s) | **g** |
| One photo | **c** |

**Goal:** ~100–150 photos each of vivek and praney + ~100 background.
- Person photos: stand **1–2 m** in front, upper body visible, turn head a little, step closer/farther. Do it in the **exam classroom**.
- Background photos: **you must NOT be in the picture** (stand behind the drone). Classmates in the picture are fine (good, even).
- Photos are saved in `exam\dataset_raw\<name>\`.

## 2b. Extra photos from a phone (optional)

- Phone **sideways**, chest height, 1–2 m, upper body. Praney with glasses.
- Copy .jpg photos into `exam\dataset_raw\vivek\` or `exam\dataset_raw\praney\` (make the folder if missing, all lowercase).
- iPhone: Settings → Camera → Formats → **Most Compatible** (so photos are .jpg, not .HEIC).
- From a phone **video** (20–30 s):
```
.venv2\Scripts\python.exe exam\import_video.py C:\path\to\video.mp4 praney
```
Keep Tello photos as the majority (the exam uses the Tello camera).

---

## 3. Recognition model: vivek / praney / other person (no drone needed)

We recognise the **face** (not clothes – with clothes the model called a classmate in a
light-blue hoodie "praney"). For every frame:
1. **Face detector** (`models\face_detector.onnx`, OpenCV YuNet) finds every face.
2. Our trained **identity model** (`models\identity.pt`, YOLO26 classifier) looks at each face
   (+ hair, glasses) → **vivek / praney / other** with its **raw confidence** (the number shown).
3. **Double check** (`models\face_recognizer.onnx` + `models\face_gallery.npz`, OpenCV SFace):
   the face must also look like that member's saved photos, otherwise → **other person**.
4. `yolo26n.pt` finds people whose face can't be seen → grey **other person** box.

Photo folders (your own photos):
```
exam\dataset_raw\Train\Vivek\   exam\dataset_raw\Train\praney\    <- training photos
exam\dataset_raw\Test\Vivek\    exam\dataset_raw\Test\praney\     <- test photos (never trained on)
exam\dataset_raw\background\                                     <- classroom WITHOUT you two
```

**A. Convert phone photos (HEIC → JPG, turned upright)**
```
.venv2\Scripts\python.exe -m pip install pillow-heif
.venv2\Scripts\python.exe exam\convert_photos.py
```
→ `exam\dataset_jpg\...` (train / test / background)

**B. Build the identity dataset (face crops + face fingerprints)**
```
.venv2\Scripts\python.exe exam\build_identity_dataset.py
```
→ `exam\identity_dataset\{train,val,test}\{vivek,praney,other}` and `exam\models\face_gallery.npz`.
**Check the crops:** open `identity_dataset\train\vivek` and `\praney`. If someone else's face is in
there, move that file into `other\`.

**C. Train** (GPU: few minutes, CPU: 15–20 min)
```
.venv2\Scripts\python.exe exam\train_identity.py
```
→ `exam\models\identity.pt` and prints the TEST accuracy (current: 99.4% on face crops).

**D. Test on the Test photos**
```
.venv2\Scripts\python.exe exam\test_identity.py
```
Current model (21 Nov): **praney 30/30, vivek 49/57, and no wrong names**. The misses are photos where the
face is dark, turned away or tiny → shown as "other person" (the safe answer: a wrong name = 0 points).
On our own phone videos: vivek standing facing the camera = **vivek 1.00 in 361/367 frames**; classmates
(also in light-blue hoodies) = "other person".

**E. Label a video or photos** (box + `vivek 0.97` / `praney 0.95` / `other person`)
```
.venv2\Scripts\python.exe exam\identify.py exam\videos\myvideo.mp4
.venv2\Scripts\python.exe exam\identify.py exam\dataset_jpg\test\praney
```
→ labelled video / photos in `exam\identify_results\`, plus a summary (frames per person, average confidence).
Phone 4K videos are shrunk to 1280 px automatically.
Settings at the top of `identify.py`: `MIN_NAME_CONF` (0.60), `MIN_SIMILARITY` (0.50, the double check), `MIN_FACE`.

**On exam day the face must be visible:** target looks at the drone; hover at face height, 1–2 m away.

**Adding more photos later:** put them in `dataset_raw`, then run A → B → C → D again.

**Moving to another laptop:** `git pull` does NOT bring photos (they're excluded from git). Copy `exam\dataset_raw` with a USB stick / zip.

---

## 4. Fly + recognise live: the FLIGHT CONSOLE (use this in the exam)

```
.venv2\Scripts\python.exe exam\flight_console.py
```
One window: **live Tello video with boxes** (`vivek 0.97` / `praney 0.95` / `other person`),
battery / height / temperature, fly buttons, a Tello command box and a log.
**Click the video once** so the window gets your keys.

| Key / button | What it does |
|---|---|
| **t** / **l** | take off / land |
| hold **w s a d** | forward / back / left / right (let go = hover) |
| hold **r f** / **q e** | up / down / turn left / turn right |
| **space** | hover |
| **x** (red button) | EMERGENCY motors off – only if about to hit a person |
| **p** | save photo now (best member frame of the last 3 s) |
| **n** | new flight (allow a new auto-save) |
| command box | type `battery?`, `temp?`, `up 30`, `cw 90` + Enter |

- **SEEN:** line shows the best member now and the exam score (`vivek 0.97 (score 117)`). Orange = below 0.80 → get closer / hold steady.
- **Auto-save:** same member 3 times in a row with confidence **≥ 0.80** (score ≥ 100) → keeps the clearest frame for 1 s → saves to `exam\exam_photos\` → re-opens it → **SAVED OK** (green).
- Can't reach 0.80? Press **p** – it still saves the raw confidence (never changed).
- Exam flight: take off → fly the ㄇ route → hover facing the person at face height, 1–2 m away → wait for **SAVED OK** → land (**l**). Press **n** before the next flight.

Settings at the top of `flight_console.py`: `SPEED`, `SAVE_MIN_CONF` (0.80), `STABLE_FRAMES`, `BEST_WINDOW`.
Backup (simple OpenCV window, same recognition): `.venv2\Scripts\python.exe exam\exam.py`

---

## 5. Exam day checklist

- [ ] Batteries charged; drone cool
- [ ] Same clothes; Praney wears glasses
- [ ] `exam\check_setup.py` prints **ALL GOOD**; `01_connect_test.py` shows video
- [ ] In `exam\models\`: `identity.pt`, `face_detector.onnx`, `face_recognizer.onnx`, `face_gallery.npz`; and `yolo26n.pt` in the UAV folder
- [ ] Laptop on TELLO Wi-Fi
- [ ] Everyone knows: **space = hover, l = land, x = emergency**; stop at once if the observer/teacher says stop
- [ ] 0 points if: hit a person, crash, leave the area, wrong name, photo not saved

## 6. After the exam – hand in

```
.venv2\Scripts\python.exe exam\make_submission.py
```
Creates `submission_<date>.zip` in the UAV folder (code + models + exam photos).

---

## How the exam program works (for the oral check)

1. **Load the models first.** `flight_console.py` (through `identify.py`) loads the face detector, our `identity.pt` (names faces), the face recognizer + gallery (double check) and `yolo26n.pt` (finds people), and runs them once to warm up.
2. **Connect to the Tello.** Connects over the Tello Wi-Fi, shows the battery, turns on the live video stream. Recognition runs in a background thread so the window and flying never freeze.
3. **Every video frame:** converts colours RGB → BGR, finds every face, crops face + hair, and our classifier says vivek / praney / other. The face must also be similar (≥ 0.50) to that member's saved photos.
4. **Draw the result.** Box on every face: `vivek 0.912` with the **raw** classifier confidence (never changed), or `other person` if not sure (below 0.60 or fails the double check). People without a visible face get a grey `other person` box. Each member is shown at most once per frame.
5. **Fly with the keyboard.** Every 50 ms it sends `send_rc_control` with the held key's movement; no key = 0,0,0,0 so the drone hovers.
6. **Avoid wrong names, aim for a high score.** Only auto-saves once the **same name** appears 3 times in a row with confidence ≥ 0.80.
7. **Pick the clearest frame.** For 1 second it keeps the frame with the highest raw confidence, then saves it to `exam_photos/` with name, confidence and time in the filename.
8. **Check the photo.** Re-opens the saved file to make sure it's valid, then shows `SAVED OK`.
9. **Safety.** Closing the window lands the drone and stops the video; x = emergency motors off.

**How the model was made:** our phone + Tello photos → `convert_photos.py` (HEIC → JPG) → `build_identity_dataset.py` found every face with OpenCV YuNet (biggest face = the member named by the folder, classmates' faces = other) and saved SFace fingerprints of the member faces → `train_identity.py` fine-tuned the YOLO26n **classifier** (`yolo26n-cls.pt`) on 3 classes of face crops → tested on the separate Test photos and our own videos.
