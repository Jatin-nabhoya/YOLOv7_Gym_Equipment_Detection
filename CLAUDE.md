# CLAUDE.md — Project Context for AI Assistants

## What This Project Is

Custom object detection pipeline that fine-tunes YOLOv7 to detect 5 gym equipment classes (dumbbell, barbell, kettlebell, resistance_band, pull_up_bar) in images. Academic project for the Deep Learning course at UNH (Spring 2026), team of 3.

The entire workflow is notebook-based — there are no standalone Python training scripts.

## Repository Layout

```
YOLOv7_Gym_Equipment_Detection/
├── notebooks/
│   ├── 01_phase1_baseline.ipynb    # data prep, augmentation, baseline inference
│   ├── 02_phase2_finetuning.ipynb  # Optuna HP search + 2-phase fine-tuning
│   └── archive/                    # older Phase 2 iterations (kept for reference)
│       ├── Phase2_v1.0.ipynb       # frozen-backbone only, 5 epochs
│       └── Phase2_v2.0.ipynb       # full fine-tuning without Optuna
├── dataset/
│   ├── images/  train(162) / val(20) / test(21) / all(203)
│   ├── labels/  train / val / test / all     (YOLO .txt format)
│   ├── classes.txt
│   └── data.yaml                   # dataset config — path relative to yolov7/
├── weights/                        # gitignored — place model files here
│   └── .gitkeep
├── docs/
│   ├── proposal.docx
│   ├── report.pdf / report.docx    # final project report
│   ├── GymEquipment_phase1.pdf     # Phase 1 PDF export
│   ├── Clean.pdf
│   ├── colab_exports/              # Colab PDF exports (reference only)
│   └── figures/                    # all visualisation PNGs
│       ├── phase1_*.png            # baseline inference, augmentation samples
│       └── phase3_*.png            # confusion matrix, training curves, per-class AP, etc.
├── yolov7/                         # gitignored — clone of WongKinYiu/yolov7
├── requirements.txt
├── README.md
└── CLAUDE.md
```

## Key Path Conventions

All notebooks live in `notebooks/` and assume:
- `../yolov7` — YOLOv7 framework clone (project root)
- `../dataset` — dataset root (train/val/test splits)
- `../weights/best.pt` — trained model checkpoint output
- `YOLOV7_DIR = os.path.abspath('../yolov7')` — defined at top of each setup cell

The `yolov7/` clone is a **nested git repo** (gitignored). Notebooks auto-clone it on first run:
```python
if not os.path.exists(YOLOV7_DIR):
    !git clone https://github.com/WongKinYiu/yolov7.git {YOLOV7_DIR}
```

`dataset/data.yaml` uses `path: ../dataset` (relative to `yolov7/`) — this is the YOLOv7-native format for `train.py --data`.

## Running the Notebooks

Run in order from the `notebooks/` directory (or open in Jupyter/Colab):

1. `01_phase1_baseline.ipynb` — installs deps, builds dataset split, runs pretrained COCO inference as baseline
2. `02_phase2_finetuning.ipynb` — Optuna HP search (30 trials), then 2-phase training (frozen backbone → full fine-tune), saves `../weights/best.pt`

Notebooks are CPU-compatible (Mac). GPU will run faster but is not required.

## What NOT to Change

- `dataset/` folder structure — YOLO format requires exactly `images/train`, `labels/train` etc.
- `dataset/classes.txt` class order — must match label indices in `.txt` files
- The `YOLOV7_DIR` / `DATASET_DIR` variables at the top of each notebook setup cell — these are the single source of truth for all path references downstream

## Dataset

- 203 total images, 5 classes, custom-collected gym equipment
- Train/val/test split: 162 / 20 / 21
- Labels in YOLO format: `class x_center y_center width height` (normalized)
- 27 spare images in `spare data /remaining_images/` (not in any split)
- Phase 1 also keeps a local copy in `phase 1/dataset/` (legacy, not used by current notebooks)

## Results Summary

| Stage | mAP@0.5 | Notes |
|-------|---------|-------|
| Baseline (COCO pretrained, no fine-tuning) | 0.0005 | Expected — wrong classes |
| Phase 2 v1 (frozen backbone, 5 epochs) | ~0.20 | archive/Phase2_v1.0.ipynb |
| Phase 2 v2 (full fine-tuning) | ~0.45 | archive/Phase2_v2.0.ipynb |
| Phase 2 v3 (Optuna + 2-phase, 50 epochs) | best so far | 02_phase2_finetuning.ipynb |

## Common Issues

- `attempt_load` fails: yolov7.pt not in `yolov7/` — re-run the setup cell to download it
- `ModuleNotFoundError: models`: `YOLOV7_DIR` not set — run cells in order from the top
- `DATASET_DIR` not found: `../dataset` doesn't exist — ensure you're running the notebook from `notebooks/`
- `weights/best.pt` not found for evaluation: run training first, or copy an existing `best.pt` into `weights/`
