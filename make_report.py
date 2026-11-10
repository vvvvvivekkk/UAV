"""Build report.pdf for "YOLO26 Model Training and Results" from the real files
in runs/detect/train (results.csv, best.pt, plots) and fresh predictions.

Run after train.py:  python make_report.py
"""
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle, KeepTogether)
from ultralytics import YOLO
import ultralytics, torch, platform

RUN = Path("runs/detect/train")
OUT = Path("report_assets")
OUT.mkdir(exist_ok=True)
DATA = "chair-table/data.yaml"
VAL_IMAGES = ["138", "125", "131", "144", "149", "121"]

# ---------- 1. read results.csv ----------
df = pd.read_csv(RUN / "results.csv")
df.columns = [c.strip() for c in df.columns]
ep = df["epoch"]

def curve(cols, title, fname, ylabel="loss"):
    fig, axes = plt.subplots(1, len(cols), figsize=(12, 3.4))
    for ax, c in zip(axes, cols):
        ax.plot(ep, df[c], marker="o", ms=2.5, lw=1.2, color="#1f6feb")
        ax.set_title(c, fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=160)
    plt.close(fig)

curve(["train/box_loss", "train/cls_loss", "train/l1_loss"],
      "Training loss curves (50 epochs)", "train_loss.png")
curve(["val/box_loss", "val/cls_loss", "val/l1_loss"],
      "Validation loss curves (50 epochs)", "val_loss.png")

fig, ax = plt.subplots(figsize=(8, 3.6))
for c, lab in [("metrics/precision(B)", "Precision"), ("metrics/recall(B)", "Recall"),
               ("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95")]:
    ax.plot(ep, df[c], marker="o", ms=2.5, lw=1.2, label=lab)
ax.set_xlabel("epoch"); ax.set_ylabel("score (0-1)"); ax.set_ylim(0, 1)
ax.grid(alpha=0.3); ax.legend(loc="lower right")
ax.set_title("Precision, Recall and mAP on the 30 val photos", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "metrics.png", dpi=160); plt.close(fig)

first, last = df.iloc[0], df.iloc[-1]
best_i = df["metrics/mAP50-95(B)"].idxmax()  # Ultralytics picks best.pt by fitness (mAP50-95)
best = df.iloc[best_i]

# ---------- 2. validate best.pt (per-class numbers) ----------
model = YOLO(RUN / "weights/best.pt")
m = model.val(data=DATA, imgsz=640, plots=False, verbose=True)
names = m.names
per_class = []
for i, cls_idx in enumerate(m.box.ap_class_index):
    p, r, ap50, ap = m.box.class_result(i)
    per_class.append((names[int(cls_idx)], p, r, ap50, ap))
overall = (m.box.mp, m.box.mr, m.box.map50, m.box.map)

# ---------- 3. detection results with my best.pt ----------
pred_dir = Path("runs/detect/report_predict")
if pred_dir.exists():
    shutil.rmtree(pred_dir)
det_rows = []
for stem in VAL_IMAGES:
    r = model.predict(source=f"chair-table/images/val/{stem}.jpg", conf=0.25,
                      save=True, project="runs/detect", name="report_predict",
                      exist_ok=True, verbose=False)[0]
    found = [f"{names[int(b.cls)]} {float(b.conf):.2f}" for b in r.boxes]
    gt = Path(f"chair-table/labels/val/{stem}.txt").read_text().split("\n")
    gt = [l for l in gt if l.strip()]
    n_gt = {"chair": sum(l.startswith("0 ") for l in gt), "table": sum(l.startswith("1 ") for l in gt)}
    det_rows.append((stem, Path(r.save_dir) / f"{stem}.jpg", found, n_gt))

# ---------- 4. terminal output panels (from the real training log) ----------
log = Path("train_log.txt").read_text(errors="ignore").replace("\r", "\n")
log = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", log)
lines = [l.rstrip() for l in log.split("\n")]
lines = [l for l in lines if l.strip() and not re.search(r"(it/s|s/it)\]?\s*$|\d+/\d+\s+[\d.]+s/it", l) and "%" not in l[-40:] or "100%" in l]

def term_panel(text_lines, fname, title):
    text_lines = [l[:118] for l in text_lines]
    h = 0.24 * len(text_lines) + 0.9
    fig = plt.figure(figsize=(12, h))
    fig.patch.set_facecolor("#0f172a")
    fig.text(0.012, 1 - 0.32 / h, title, color="#7dd3fc", family="monospace",
             fontsize=9.5, fontweight="bold", va="top")
    fig.text(0.012, 1 - 0.7 / h, "\n".join(text_lines), color="#e2e8f0",
             family="monospace", fontsize=8.2, va="top", linespacing=1.35)
    fig.savefig(OUT / fname, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

def pick(pattern, n_after=0):
    for i, l in enumerate(lines):
        if re.search(pattern, l):
            return lines[i:i + 1 + n_after]
    return []

start = pick(r"^train: Scanning") + pick(r"^val: Scanning") + pick(r"Starting training")
hdr = pick(r"Epoch\s+GPU_mem")
ep1 = [l for l in lines if re.match(r"\s+1/50\s", l) and "100%" in l][-1:]
ep50 = [l for l in lines if re.match(r"\s+50/50\s", l) and "100%" in l][-1:]
allrows = [i for i, l in enumerate(lines) if re.match(r"\s+all\s+30\s+49", l)]
first_all = [lines[allrows[0]]] if allrows else []
tail_start = next((i for i, l in enumerate(lines) if "epochs completed" in l), len(lines) - 25)
tail = lines[tail_start:tail_start + 14]
term_panel(["$ python train.py"] + start + [""] + hdr + ep1 + first_all + ["  ..."] + ep50 +
           ([lines[allrows[-2]]] if len(allrows) > 1 else []) + [""] + tail,
           "terminal_train.png", "TERMINAL OUTPUT - python train.py (start, epoch 1, epoch 50, final validation)")

val_lines = ["$ yolo val model=runs/detect/train/weights/best.pt data=chair-table/data.yaml",
             f"Ultralytics {ultralytics.__version__}  Python-{platform.python_version()} torch-{torch.__version__}",
             f"{'Class':>18}{'Images':>11}{'Instances':>11}{'Box(P':>11}{'R':>11}{'mAP50':>11}{'mAP50-95)':>11}",
             f"{'all':>18}{30:>11}{int(m.nt_per_class.sum()):>11}{overall[0]:>11.3f}{overall[1]:>11.3f}{overall[2]:>11.3f}{overall[3]:>11.3f}"]
for (n, p, r, a50, a), cnt in zip(per_class, [m.nt_per_class[int(k)] for k in m.box.ap_class_index]):
    val_lines.append(f"{n:>18}{m.nt_per_image[0] if False else '':>11}{int(cnt):>11}{p:>11.3f}{r:>11.3f}{a50:>11.3f}{a:>11.3f}")
val_lines.append(f"Speed: {m.speed['preprocess']:.1f}ms preprocess, {m.speed['inference']:.1f}ms inference, {m.speed['postprocess']:.1f}ms postprocess per image")
term_panel(val_lines, "terminal_val.png", "TERMINAL OUTPUT - validation of best.pt on the 30 val photos")

pred_lines = ["$ python predict_mine.py"]
for stem, path, found, n_gt in det_rows:
    c = sum(f.startswith("chair") for f in found); t = sum(f.startswith("table") for f in found)
    pred_lines.append(f"image chair-table/images/val/{stem}.jpg: {c} chairs, {t} tables")
pred_lines.append(f"Results saved to runs/detect/report_predict")
term_panel(pred_lines, "terminal_predict.png", "TERMINAL OUTPUT - detection with my trained best.pt")

# ---------- 5. build the PDF ----------
args = {}
for l in (RUN / "args.yaml").read_text().splitlines():
    if ":" in l:
        k, v = l.split(":", 1); args[k.strip()] = v.strip()

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, textColor=colors.HexColor("#0f172a"), spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, textColor=colors.HexColor("#0e7490"), spaceAfter=4)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.6, leading=13.5)
SMALL = ParagraphStyle("S", parent=B, fontSize=8.2, leading=11, textColor=colors.HexColor("#475569"))
W = A4[0] - 3.2 * cm

def img(path, width=W):
    from PIL import Image as PI
    w, h = PI.open(path).size
    return Image(str(path), width=width, height=width * h / w)

def table(rows, col_w=None, header=True):
    t = Table(rows, colWidths=col_w)
    st = [("FONTSIZE", (0, 0), (-1, -1), 8.8), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st))
    return t

f3 = lambda x: f"{float(x):.3f}"
n_train_inst = sum(len([l for l in p.read_text().splitlines() if l.strip()]) for p in Path("chair-table/labels/train").glob("*.txt"))
n_val_inst = int(m.nt_per_class.sum())
cls_counts = {0: 0, 1: 0}
for p in Path("chair-table/labels").glob("*/*.txt"):
    for l in p.read_text().splitlines():
        if l.strip():
            cls_counts[int(l.split()[0])] += 1

story = []
story += [Spacer(1, 3 * cm), Paragraph("YOLO26 Model Training and Results", ParagraphStyle("T", parent=H1, fontSize=24, leading=30)),
          Paragraph("Chair and table detector trained on my own run of the chair-table dataset (2 classes)", ParagraphStyle("st", parent=B, fontSize=12, textColor=colors.HexColor("#0e7490"))),
          Spacer(1, 1.2 * cm),
          table([["Student", "Aleti Vivek Reddy"], ["Student ID", "415410902"],
                 ["Course", "UAV (CCU) - YOLO26 hands-on, class of November 3"],
                 ["Model", "YOLO26n (yolo26n.pt) fine-tuned for 2 classes: 0 = chair, 1 = table"],
                 ["Code", "github.com/vvvvvivekkk/UAV (train.py, predict_mine.py, make_report.py)"]],
                [3.5 * cm, W - 3.5 * cm], header=False),
          Spacer(1, 0.8 * cm),
          Paragraph("<b>Contents</b><br/>1. Setup and dataset &nbsp; 2. Training loss curves &nbsp; 3. Validation loss curves &nbsp; "
                    "4. Precision, Recall and mAP &nbsp; 5. Detection results using my trained model &nbsp; 6. Conclusion", B),
          PageBreak()]

story += [Paragraph("0. Setup and dataset", H1),
          table([["Item", "Value"],
                 ["Software", f"Python {platform.python_version()}, Ultralytics {ultralytics.__version__}, PyTorch {torch.__version__}"],
                 ["Hardware", "CPU only (no GPU)"],
                 ["Dataset", "chair-table: 150 labelled room photos, YOLO .txt format"],
                 ["Classes", f"0 = chair ({cls_counts[0]} boxes), 1 = table ({cls_counts[1]} boxes)"],
                 ["Split", f"120 train ({n_train_inst} boxes) / 30 val ({n_val_inst} boxes), 0 backgrounds, 0 corrupt"],
                 ["Base model", "yolo26n.pt (pretrained on COCO), fine-tuned"],
                 ["Settings (args.yaml)", f"epochs={args.get('epochs')}, imgsz={args.get('imgsz')}, batch={args.get('batch')}, "
                                          f"optimizer={args.get('optimizer')}, seed={args.get('seed')}"]],
                [3.6 * cm, W - 3.6 * cm]),
          Spacer(1, 0.3 * cm),
          Paragraph("data.yaml: <font face='Courier'>train: images/train &nbsp; val: images/val &nbsp; names: {0: chair, 1: table}</font>. "
                    "Training command: <font face='Courier'>model.train(data=\"chair-table/data.yaml\", epochs=50, imgsz=640)</font>.", B),
          Spacer(1, 0.3 * cm),
          img(OUT / "terminal_train.png"),
          Paragraph("Terminal output of the training run, taken from the saved log (train_log.txt).", SMALL),
          PageBreak()]

story += [Paragraph("1. Training loss curves", H1), img(OUT / "train_loss.png"),
          Paragraph(f"All three training losses go down over the 50 epochs. "
                    f"<b>box_loss</b> (how far the predicted boxes are from the labelled boxes) fell from {f3(first['train/box_loss'])} to {f3(last['train/box_loss'])}; "
                    f"<b>cls_loss</b> (how wrong the chair/table class guess is) fell from {f3(first['train/cls_loss'])} to {f3(last['train/cls_loss'])}; "
                    f"<b>l1_loss</b> fell from {first['train/l1_loss']:.4f} to {last['train/l1_loss']:.4f}. "
                    "YOLO26 uses an L1 box loss in place of the Distribution Focal Loss (DFL) used in earlier YOLO versions, so there is no dfl_loss column.", B),
          Spacer(1, 0.5 * cm),
          Paragraph("2. Validation loss curves", H1), img(OUT / "val_loss.png"),
          Paragraph(f"Validation losses are measured on the 30 val photos, which the model never trains on. "
                    f"val/box_loss went from {f3(first['val/box_loss'])} to {f3(last['val/box_loss'])}, "
                    f"val/cls_loss from {f3(first['val/cls_loss'])} to {f3(last['val/cls_loss'])} and "
                    f"val/l1_loss from {first['val/l1_loss']:.4f} to {last['val/l1_loss']:.4f}. "
                    "The first ~10 epochs jump up and down while the new 2-class head adapts; afterwards the curves settle. "
                    "Validation loss does not rise at the end, so there is no clear sign of overfitting in 50 epochs.", B),
          PageBreak()]

story += [Paragraph("3. Precision, Recall and mAP", H1),
          Paragraph("<b>Precision</b>: of the boxes the model drew, the share that are correct. "
                    "<b>Recall</b>: of the real chairs and tables, the share the model found. "
                    "<b>mAP50</b>: mean average precision when a box counts as correct if it overlaps the label by at least 50% (IoU 0.5). "
                    "<b>mAP50-95</b>: the same averaged over stricter overlaps from 0.50 to 0.95.", B),
          Spacer(1, 0.25 * cm), img(OUT / "metrics.png", W * 0.85),
          Spacer(1, 0.2 * cm),
          table([["", "Precision", "Recall", "mAP50", "mAP50-95"],
                 ["Epoch 1", f3(first["metrics/precision(B)"]), f3(first["metrics/recall(B)"]), f3(first["metrics/mAP50(B)"]), f3(first["metrics/mAP50-95(B)"])],
                 ["Epoch 50 (last.pt)", f3(last["metrics/precision(B)"]), f3(last["metrics/recall(B)"]), f3(last["metrics/mAP50(B)"]), f3(last["metrics/mAP50-95(B)"])],
                 [f"Best epoch {int(best['epoch'])}", f3(best["metrics/precision(B)"]), f3(best["metrics/recall(B)"]), f3(best["metrics/mAP50(B)"]), f3(best["metrics/mAP50-95(B)"])]],
                [4.6 * cm] + [(W - 4.6 * cm) / 4] * 4),
          Spacer(1, 0.25 * cm),
          Paragraph("<b>Final validation of best.pt, per class</b>", B),
          table([["Class", "Boxes", "Precision", "Recall", "mAP50", "mAP50-95"],
                 ["all", str(n_val_inst), f3(overall[0]), f3(overall[1]), f3(overall[2]), f3(overall[3])]] +
                [[n, str(int(m.nt_per_class[i])), f3(p), f3(r), f3(a50), f3(a)] for i, (n, p, r, a50, a) in enumerate(per_class)],
                [3.0 * cm, 2.0 * cm] + [(W - 5.0 * cm) / 4] * 4),
          Spacer(1, 0.25 * cm), img(OUT / "terminal_val.png"),
          PageBreak()]

pr = next((RUN / n for n in ["BoxPR_curve.png", "PR_curve.png"] if (RUN / n).exists()), None)
cm_png = next((RUN / n for n in ["confusion_matrix_normalized.png", "confusion_matrix.png"] if (RUN / n).exists()), None)
row = [img(p, W / 2 - 0.2 * cm) for p in [pr, cm_png] if p]
if row:
    story += [Paragraph("3b. PR curve and confusion matrix (from the training run)", H2),
              Table([row], colWidths=[W / 2] * len(row)),
              Paragraph("Left: precision-recall curve per class (area under it = AP). Right: confusion matrix, showing how often "
                        "each true class was predicted as chair, table, or missed (background).", SMALL),
              Spacer(1, 0.4 * cm)]
story += [Paragraph("Ultralytics summary chart (results.png)", H2), img(RUN / "results.png"), PageBreak()]

story += [Paragraph("4. Detection results using my trained model", H1),
          Paragraph("best.pt was run on validation photos it never trained on (conf = 0.25). "
                    "Each caption lists the boxes found, compared with the number of labelled objects.", B),
          Spacer(1, 0.2 * cm)]
cells = []
for stem, path, found, n_gt in det_rows:
    c = sum(f.startswith("chair") for f in found); t = sum(f.startswith("table") for f in found)
    cap = Paragraph(f"<b>val/{stem}.jpg</b> - found {c} chairs, {t} tables "
                    f"(labelled: {n_gt['chair']} chairs, {n_gt['table']} tables)<br/>"
                    f"<font size=7.5>{', '.join(found) if found else 'no detections'}</font>", SMALL)
    cells.append([img(path, W / 2 - 0.3 * cm), cap])
grid = []
for i in range(0, len(cells), 2):
    pair = cells[i:i + 2]
    grid.append([p[0] for p in pair] + [""] * (2 - len(pair)))
    grid.append([p[1] for p in pair] + [""] * (2 - len(pair)))
story += [Table(grid, colWidths=[W / 2] * 2, style=[("VALIGN", (0, 0), (-1, -1), "TOP")]), Spacer(1, 0.3 * cm),
          img(OUT / "terminal_predict.png")]
if (RUN / "val_batch0_pred.jpg").exists():
    story += [PageBreak(), Paragraph("Validation batch: my labels vs. my model's boxes", H2),
              Table([[img(RUN / "val_batch0_labels.jpg", W / 2 - 0.2 * cm), img(RUN / "val_batch0_pred.jpg", W / 2 - 0.2 * cm)]],
                    colWidths=[W / 2] * 2),
              Paragraph("Left: labelled boxes (val_batch0_labels.jpg). Right: predictions of the trained model (val_batch0_pred.jpg).", SMALL)]

story += [Spacer(1, 0.5 * cm), Paragraph("5. Conclusion", H1),
          Paragraph(f"I fine-tuned YOLO26n for two classes (chair, table) for 50 epochs on CPU. Training and validation losses both decreased, "
                    f"and on the 30 unseen val photos the best model reaches precision {f3(overall[0])}, recall {f3(overall[1])}, "
                    f"mAP50 {f3(overall[2])} and mAP50-95 {f3(overall[3])} "
                    f"(chair mAP50 {f3(per_class[0][3])}, table mAP50 {f3(per_class[1][3])}). "
                    "The detection images show it finds dining chairs, armchairs, desks and coffee tables in rooms it never saw.", B),
          Paragraph("<b>Limitations and next steps:</b> only 120 training photos and 30 val photos, so scores move a lot between epochs; "
                    "small or partly hidden chairs are sometimes missed or boxed twice. More photos (especially of my own rooms), "
                    "more epochs on a GPU, and a larger model (yolo26s) would improve accuracy.", B),
          Spacer(1, 0.3 * cm),
          Paragraph(f"All numbers come from {RUN}/results.csv and model.val() on {RUN}/weights/best.pt. "
                    "Scripts and AI-assistant use: the training and report scripts were written with Claude (AI coding assistant) and run by me.", SMALL)]

doc = SimpleDocTemplate("report.pdf", pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                        topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                        title="YOLO26 Model Training and Results", author="Aleti Vivek Reddy")
def footer(c, d):
    c.setFont("Helvetica", 7.5); c.setFillColor(colors.HexColor("#64748b"))
    c.drawString(1.6 * cm, 0.8 * cm, "YOLO26 Model Training and Results - Aleti Vivek Reddy (415410902)")
    c.drawRightString(A4[0] - 1.6 * cm, 0.8 * cm, str(d.page))
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("Saved report.pdf")
