r"""
CHECK SETUP - is this laptop ready for the drone exam? (no drone needed)

Checks: all Python packages, all model files, and runs the recogniser once.
Run (from the UAV folder):
  .venv2\Scripts\python.exe exam\check_setup.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent
problems = []


def check(label, fn):
    try:
        info = fn()
        print(f"  OK   {label}" + (f"  ({info})" if info else ""))
    except Exception as ex:
        print(f"  FAIL {label}: {ex}")
        problems.append(label)


print(f"Python {sys.version.split()[0]}  ({sys.executable})\n\nPackages:")
check("tkinter (window)", lambda: __import__("tkinter").TkVersion)
check("opencv-python", lambda: __import__("cv2").__version__)
check("numpy", lambda: __import__("numpy").__version__)
check("pillow", lambda: __import__("PIL").__version__)
check("pillow-heif", lambda: __import__("pillow_heif").__version__)
check("djitellopy (Tello)", lambda: __import__("djitellopy") and "")
check("ultralytics (YOLO26)", lambda: __import__("ultralytics").__version__)
check("torch", lambda: (lambda t: f"{t.__version__}, GPU: {t.cuda.is_available()}")(__import__("torch")))
check("OpenCV face tools", lambda: (__import__("cv2").FaceDetectorYN, __import__("cv2").FaceRecognizerSF) and "")

print("\nModel files:")
for f in [HERE.parent / "yolo26n.pt", HERE / "models" / "identity.pt", HERE / "models" / "face_detector.onnx",
          HERE / "models" / "face_recognizer.onnx", HERE / "models" / "face_gallery.npz"]:
    check(str(f.relative_to(HERE.parent)), lambda f=f: f"{f.stat().st_size / 1e6:.1f} MB")

if not problems:
    print("\nRecogniser test:")
    def run():
        import time
        import cv2
        import numpy as np
        sys.path.insert(0, str(HERE))
        import identify as ident
        ident.identify(np.zeros((720, 960, 3), np.uint8))
        img = cv2.imread(str(HERE.parent / "zidane.jpg")) if (HERE.parent / "zidane.jpg").exists() \
            else np.zeros((720, 960, 3), np.uint8)
        t = time.time()
        res = ident.identify(cv2.resize(img, (960, 720)))
        return f"{len(res)} people found in {time.time() - t:.2f} s: " + ", ".join(r[0] for r in res)
    check("identify.py runs", run)

if problems:
    print("\nNOT READY. Problems:", ", ".join(problems))
    print("Fix: run setup.bat again (normal internet). Missing model files: git pull or copy them.")
else:
    print("\nALL GOOD. Next: connect to the TELLO-XXXXXX Wi-Fi and run:")
    print("  .venv2\\Scripts\\python.exe exam\\flight_console.py")
