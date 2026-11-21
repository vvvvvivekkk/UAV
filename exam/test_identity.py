r"""
TEST - check the identity model on photos it never trained on.

For every photo in exam\dataset_jpg\test\vivek and \praney it runs the full
pipeline (find faces -> name them) and checks that the right member is found
and the other member is NOT wrongly named in that photo.

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
        res = ident.identify(ident.shrink(img))
        labels = [r[0] for r in res]
        others = [m for m in ident.MEMBERS if m != member]
        if member in labels and not any(m in labels for m in others):
            ok += 1
            confs.append(next(r[1] for r in res if r[0] == member))
        else:
            wrong.append((f.name, ", ".join(labels) or "nobody found"))
    total += len(files)
    correct += ok
    avg = sum(confs) / len(confs) if confs else 0
    print(f"{member:7s}: {ok}/{len(files)} correct ({100 * ok / max(1, len(files)):.0f}%), average confidence when correct {avg:.2f}")
    for name, got in wrong:
        print(f"    wrong: {name} -> {got}")

print(f"\nTOTAL: {correct}/{total} correct ({100 * correct / max(1, total):.0f}%)")
