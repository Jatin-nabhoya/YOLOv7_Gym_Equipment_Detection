# YOLOv7 Gym Equipment Detection

Fine-tuned YOLOv7 object detection pipeline for identifying gym equipment in images. Built as part of the Deep Learning course at the University of New Hampshire (UNH), Spring 2026.

---

## Classes

| ID | Equipment |
|----|-----------|
| 0 | Dumbbell |
| 1 | Barbell |
| 2 | Kettlebell |
| 3 | Resistance Band |
| 4 | Pull-up Bar |

---

## Project Structure

```
YOLOv7_Gym_Equipment_Detection/
├── notebooks/
│   ├── 01_phase1_baseline.ipynb     # Data prep, augmentation, baseline inference
│   ├── 02_phase2_finetuning.ipynb   # Optuna HP search + 2-phase fine-tuning
│   └── archive/                     # Older iterations kept for reference
├── dataset/
│   ├── images/   train/ val/ test/  # 162 / 20 / 21 images
│   ├── labels/   train/ val/ test/  # YOLO-format annotations
│   ├── classes.txt
│   └── data.yaml
├── weights/                         # Model checkpoints (gitignored — not committed)
├── docs/
│   ├── proposal.docx
│   ├── report.pdf
│   └── figures/                     # Training curves, confusion matrix, etc.
├── yolov7/                          # YOLOv7 source clone (gitignored)
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/Jatin-nabhoya/YOLOv7_Gym_Equipment_Detection.git
cd YOLOv7_Gym_Equipment_Detection
```

### 2. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
```

### 3. Run the notebooks in order

Open in Jupyter or VS Code from the `notebooks/` directory:

```
notebooks/01_phase1_baseline.ipynb    → data setup, baseline evaluation
notebooks/02_phase2_finetuning.ipynb  → Optuna search + training → weights/best.pt
```

Each notebook auto-clones the YOLOv7 framework on first run. No manual setup needed beyond `pip install -r requirements.txt`.

### 4. Place model weights (optional — skip if training from scratch)

Copy a pre-trained checkpoint into `weights/`:

```
weights/best.pt    ← output of 02_phase2_finetuning.ipynb
```

---

## Methodology

### Phase 1 — Baseline

- Loaded pretrained YOLOv7 (COCO) without fine-tuning
- Confirmed near-zero mAP on gym equipment (expected — wrong domain)
- Built custom dataset loader with augmentations (flip, color jitter, mosaic)

### Phase 2 — Fine-tuning

- **Optuna hyperparameter search** (30 trials): learning rate, weight decay, batch size, augmentation strength
- **2-phase training**:
  1. Freeze backbone, train detection head (warm-up)
  2. Unfreeze all layers, full end-to-end fine-tuning (50 epochs, early stopping at patience=15)
- **TTA (Test-Time Augmentation)** at evaluation

---

## Results

| Stage | mAP@0.5 | mAP@0.5:0.95 |
|-------|---------|---------------|
| Baseline (COCO pretrained) | 0.0005 | 0.0001 |
| Phase 2 v1 (frozen backbone) | ~0.20 | — |
| Phase 2 v2 (full fine-tuning) | ~0.45 | — |
| Phase 2 v3 (Optuna + 2-phase) | best | see report |

See [`docs/report.pdf`](docs/report.pdf) for full results and figures.

---

## Dataset

- **203 images** across 5 gym equipment classes, custom-collected
- **Split**: 162 train / 20 val / 21 test
- **Format**: YOLO (normalized `class x_center y_center width height` per line)
- Dataset images and labels are gitignored due to size — contact the team for access

---

## Dependencies

See [`requirements.txt`](requirements.txt). Key packages:

- `torch` / `torchvision` >= 2.0
- `opencv-python`, `Pillow`, `albumentations`
- `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `torchmetrics`
- `optuna` (hyperparameter search)
- `PyYAML`, `tqdm`

---

## Architecture

- **Base**: YOLOv7 pretrained on COCO (WongKinYiu/yolov7)
- **Modification**: Detection head replaced for 5 custom classes
- **Framework**: PyTorch, custom training loop (not YOLOv7's `train.py`)

---

## Course Information

- **Course**: Deep Learning — University of New Hampshire (UNH)
- **Semester**: Spring 2026

---

## License

Academic use only. YOLOv7 architecture by [WongKinYiu](https://github.com/WongKinYiu/yolov7), used under its original license.
