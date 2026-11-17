# UAV Exam – Full Step-by-Step Guide (do it yourself)

Team: **vivek** (no glasses) and **praney** (glasses). Same clothes on photo day and exam day.
Always open a terminal **in the UAV folder** (`C:\Users\11020\Desktop\UAV`) and copy the commands exactly.

---

## 0. One-time setup (needs normal internet, NOT the Tello Wi-Fi)

```
py -m venv .venv2
.venv2\Scripts\python.exe -m pip install djitellopy==2.5.0 opencv-python ultralytics
```
Takes a few minutes (downloads PyTorch). Do this on every new laptop (e.g. Praney's).

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

## 3. Build the dataset (no drone needed)

```
.venv2\Scripts\python.exe exam\04_make_dataset.py
```
It prints e.g. `Photos used per class: {'vivek': 140, 'praney': 130, 'background': 100}`.
- It boxes each person automatically (with `yolo26n.pt`) and names them by folder.
- Needs **both** vivek and praney photos – otherwise the model calls everyone "vivek".
- Optional: open `exam\team_dataset` in X-AnyLabeling to check the boxes.

## 4. Train (no drone needed)

```
.venv2\Scripts\python.exe exam\05_train.py
```
CPU: about 20–60 min. Watch `mAP50` go up (closer to 1.0 = better).
Result: `exam\runs\team\weights\best.pt`.
To retrain from scratch later, delete the folders `exam\runs` and `exam\team_dataset` first.

**Moving to another laptop:** `git pull` does NOT bring photos (they're excluded from git). Copy `exam\dataset_raw` with a USB stick / zip.

---

## 5. Practice the exam

```
.venv2\Scripts\python.exe exam\exam.py
```
Click the video window. Same flying keys (t, l, w/s/a/d, r/f, q/e, space, x = emergency, ESC = land + quit).
- The screen shows the box + `vivek 0.912` (raw confidence from the model).
- **Auto-save:** when the same name is seen 5 frames in a row (≥ 0.50), it waits 1 s, saves the clearest frame to `exam\exam_photos\` and shows **SAVED OK**.
- **p** = save a photo now (backup).
- Fly the ㄇ route, hover facing the person ~1–2 m away, steady, until SAVED OK. Then land (**l**).

Settings you can change at the top of `exam.py`: `SPEED`, `MIN_CONF`, `STABLE_FRAMES`, `BEST_WINDOW`.

---

## 6. Exam day checklist

- [ ] Batteries charged; drone cool
- [ ] Same clothes; Praney wears glasses
- [ ] `.venv2` works (`01_connect_test.py` runs)
- [ ] `exam\runs\team\weights\best.pt` exists
- [ ] Laptop on TELLO Wi-Fi
- [ ] Everyone knows: **space = hover, l = land, x = emergency**; stop at once if the observer/teacher says stop
- [ ] 0 points if: hit a person, crash, leave the area, wrong name, photo not saved

## 7. After the exam – hand in

```
.venv2\Scripts\python.exe exam\make_submission.py
```
Creates `submission_<date>.zip` in the UAV folder (code + `best.pt` + `data.yaml` + exam photos).

---

## How the exam program works (for the oral check)

1. **Load the model first.** `exam.py` loads our trained YOLO26 model (`best.pt`) and runs it once on a black image to warm it up, so the first real prediction isn't slow.
2. **Connect to the Tello.** Connects over the Tello Wi-Fi, prints the battery, turns on the live video stream.
3. **Every video frame:** converts colours RGB → BGR (what OpenCV uses) and runs our model on it.
4. **Draw the result.** Takes the most confident detection and draws a box with the name and the **raw** confidence, e.g. `vivek 0.912`. We never change that number.
5. **Fly with the keyboard.** Keys send `send_rc_control`; with no key pressed it sends 0,0,0,0 so the drone hovers.
6. **Avoid wrong names.** Only saves once the **same name** appears in 5 frames in a row (confidence ≥ 0.50).
7. **Pick the clearest frame.** For 1 second it keeps the frame with the highest raw confidence, then saves it to `exam_photos/` with name, confidence and time in the filename.
8. **Check the photo.** Re-opens the saved file to make sure it's valid, then shows `SAVED OK`.
9. **Safety.** Whatever happens (even an error), the `finally` block lands the drone and closes the video.

**How the model was made:** photos with the Tello camera (+ some phone photos) → `04_make_dataset.py` boxed each person with the pretrained `yolo26n.pt`, name from the folder, 80/20 train/val → `05_train.py` fine-tuned YOLO26n on our 2 classes.
