# Exam code (UAV practical exam)

Run everything from the **UAV folder** with the existing venv:
`.venv2\Scripts\python.exe exam\<file>.py`

Before running: connect the laptop Wi-Fi to **TELLO-XXXXXX** (no internet on that Wi-Fi).
Always click on the video window before pressing keys.

| Step | File | Drone flies? | What it does |
|---|---|---|---|
| 1 | `01_connect_test.py` | No | Connect, show battery, show live video 20 s |
| 2 | `02_fly_test.py` | Yes | Fly with the keyboard while watching the video |
| 3 | `03_collect_photos.py` | No | Save training photos of each teammate |
| 4 | `04_make_dataset.py` | No | Auto-box each person, split train/val, write data.yaml |
| 5 | `05_train.py` | No | Train YOLO26 on your teammates -> `exam\runs\team\weights\best.pt` |
| 6 | `exam.py` | Yes | **The exam:** fly + recognise teammate + auto-save photo |
| 7 | `make_submission.py` | No | Zip code + model + photos to hand in |

## Flying keys (02_fly_test.py)
t take off · l land · w/s forward/back · a/d left/right · r/f up/down · q/e turn
**SPACE = hover** · **x = emergency motors off (only if about to hit a person)** · ESC = land + quit

## Safety rules
- Safety observer watches the drone; if they say **stop**, press SPACE (hover) or l (land) at once.
- No takeoff below 20% battery. Clear the area first.
- Hitting a person, crashing, or leaving the area = 0 points for that flight.

## exam.py extra keys
**p = save photo now** (backup if auto-save didn't happen). Auto-save: same name 5 frames in a row (conf >= 0.50), then the clearest frame within 1 s is saved to `exam\exam_photos\` and checked.

Team: **vivek** (key 1) and **praney** (key 2). Names are set in 03_collect_photos.py and 04_make_dataset.py (same order in both).
Tip: praney wears glasses, vivek doesn't - keep it that way for photos AND exam day (and same clothes).
See **FLOW.md** for how the program works (for the oral check).
