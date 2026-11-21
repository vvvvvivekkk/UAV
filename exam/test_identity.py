r"""
TEST - check the identity model on photos it never trained on.

For every photo in exam\dataset_jpg\test\vivek and \praney it runs the full
pipeline (find people -> name them) and checks that the MAIN person (biggest box)
gets the right name. Background test photos must give only "other person".

Run (from the UAV folder):
  .venv2\Scripts\python.exe exam\test_identity.py
"""
from pathlib import Path

import identify as ident   # same folder

TEST = ident.HERE / "dataset_jpg" / "test"
import cv2

total = correct = 0
rows = []
for member in ident.MEMBERS:
    files = sorted((TEST / member).glob("*.jpg"))
    ok = 0
    confs = []
    wrong = []
    for f in files:
        img = cv2.imread(str(f))
        res = ident.identify(img)
        if not res:
            wrong.append((f.name, "nobody found"))
            continue
        main = max(res, key=lambda r: (r[2][2] - r[2][0]) * (r[2][3] - r[2][1]))
        if main[0] == member:
            ok += 1
            confs.append(main[1])
        else:
            wrong.append((f.name, main[0]))
    total += len(files)
    correct += ok
    avg = sum(confs) / len(confs) if confs else 0
    print(f"{member:7s}: {ok}/{len(files)} correct ({100 * ok / max(1, len(files)):.0f}%), average confidence when correct {avg:.2f}")
    for name, got in wrong:
        print(f"    wrong: {name} -> {got}")

print(f"\nTOTAL: {correct}/{total} correct ({100 * correct / max(1, total):.0f}%)")
