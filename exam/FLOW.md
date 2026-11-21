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

## 3. Recognition model: vivek / praney / other person (no drone needed)

We use **2 steps**: `yolo26n.pt` finds every person (box), then our trained
**identity model** (`exam\models\identity.pt`) names each one: **vivek**, **praney** or
**other person**, with the model's raw confidence. Anyone who isn't clearly vivek or
praney (classmates, strangers) is shown as **other person**.

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

**B. Build the identity dataset (crops of every person)**
```
.venv2\Scripts\python.exe exam\build_identity_dataset.py
```
→ `exam\identity_dataset\{train,val,test}\{vivek,praney,other}`.
**Check the crops:** open `identity_dataset\train\vivek` and `\praney`. If a classmate is in there
(happens when they stood closer to the camera than you), move that file into `other\`.

**C. Train** (GPU: few minutes, CPU: 15–30 min)
```
.venv2\Scripts\python.exe exam\train_identity.py
```
→ `exam\models\identity.pt` and prints the TEST accuracy.

**D. Test on the Test photos**
```
.venv2\Scripts\python.exe exam\test_identity.py
```
Current model (21 Nov): **vivek 57/57, praney 30/30 = 100%**, no classmate wrongly named.

**E. Label a video or photos** (box + `vivek 0.97` / `praney 0.95` / `other person`)
```
.venv2\Scripts\python.exe exam\identify.py exam\videos\myvideo.mp4
.venv2\Scripts\python.exe exam\identify.py exam\dataset_jpg\test\praney
```
→ labelled video / photos in `exam\identify_results\`, plus a summary (frames per person, average confidence).
Settings at the top of `identify.py`: `MIN_NAME_CONF` (0.60: below this a name guess becomes "other person"), `PERSON_CONF`.

**Adding more photos later:** put them in `dataset_raw`, then run A → B → C → D again.

**Moving to another laptop:** `git pull` does NOT bring photos (they're excluded from git). Copy `exam\dataset_raw` with a USB stick / zip.

---

## 4. Practice the exam

```
.venv2\Scripts\python.exe exam\exam.py
```
Click the video window. Same flying keys (t, l, w/s/a/d, r/f, q/e, space, x = emergency, ESC = land + quit).
- The screen shows every person: `vivek 0.912` / `praney 0.95` (raw confidence) or `other person`.
- **Auto-save:** when the same name is seen 5 frames in a row (≥ 0.50), it waits 1 s, saves the clearest frame to `exam\exam_photos\` and shows **SAVED OK**.
- **p** = save a photo now (backup).
- Fly the ㄇ route, hover facing the person ~1–2 m away, steady, until SAVED OK. Then land (**l**).

Settings at the top of `exam.py`: `SPEED`, `STABLE_FRAMES`, `BEST_WINDOW`; name threshold `MIN_NAME_CONF` is in `identify.py`.

---

## 5. Exam day checklist

- [ ] Batteries charged; drone cool
- [ ] Same clothes; Praney wears glasses
- [ ] `.venv2` works (`01_connect_test.py` runs)
- [ ] `exam\models\identity.pt` and `yolo26n.pt` exist
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

1. **Load the models first.** `exam.py` loads `yolo26n.pt` (finds people) and our `identity.pt` (names them) and runs them once on a black image to warm up.
2. **Connect to the Tello.** Connects over the Tello Wi-Fi, prints the battery, turns on the live video stream.
3. **Every video frame:** converts colours RGB → BGR, finds every person, crops head + upper body, and our classifier says vivek / praney / other.
4. **Draw the result.** Box on every person: `vivek 0.912` with the **raw** confidence (never changed), or `other person` if not sure (below 0.60). Each member is shown at most once per frame.
5. **Fly with the keyboard.** Keys send `send_rc_control`; with no key pressed it sends 0,0,0,0 so the drone hovers.
6. **Avoid wrong names.** Only saves once the **same name** appears in 5 frames in a row (confidence ≥ 0.60).
7. **Pick the clearest frame.** For 1 second it keeps the frame with the highest raw confidence, then saves it to `exam_photos/` with name, confidence and time in the filename.
8. **Check the photo.** Re-opens the saved file to make sure it's valid, then shows `SAVED OK`.
9. **Safety.** Whatever happens (even an error), the `finally` block lands the drone and closes the video.

**How the model was made:** our phone + Tello photos → `convert_photos.py` (HEIC → JPG) → `build_identity_dataset.py` cropped every person with the pretrained `yolo26n.pt` (biggest person = the member named by the folder, classmates in background photos = other) → `train_identity.py` fine-tuned the YOLO26n **classifier** (`yolo26n-cls.pt`) on 3 classes → tested on the separate Test photos: 100% correct.
