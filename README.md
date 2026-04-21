# 🏋️ YOLOv7 Gym Equipment Detection

A custom object detection pipeline built on **YOLOv7** to identify gym equipment in images and video. This project was completed as part of the Deep Learning course at the **University of New Hampshire (UNH)**.

---

## 📌 Project Overview

This project fine-tunes the YOLOv7 architecture on a custom dataset of gym equipment images to accurately detect and localize **5 classes** of gym equipment:

| Class ID | Equipment |
|----------|-----------|
| 0 | Dumbbell |
| 1 | Barbell |
| 2 | Kettlebell |
| 3 | Resistance Band |
| 4 | Pull-up Bar |

---

## 🗂️ Project Structure

```
YOLOv7_Gym_Equipment_Detection/
├── phase 1/                     # Phase 1 — Baseline & Preprocessing
│   ├── Project_Update1.ipynb    # Phase 1 notebook (data prep, augmentation, baseline mAP)
│   ├── GymEquipment.ipynb       # Full Phase 1 notebook
│   ├── dataset/                 # Phase 1 local dataset split
│   └── yolov7/                  # YOLOv7 source (submodule)
│
├── Phase 2/                     # Phase 2 — Transfer Learning & Fine-tuning
│   ├── Phase2_v1.0.ipynb        # Training with frozen backbone
│   ├── Phase2_v2.0.ipynb        # End-to-end fine-tuning
│   └── yolov7.pt                # Pretrained YOLOv7 weights (COCO)
│
├── dataset/                     # Master dataset with YOLO annotation format
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── classes.txt
│   └── data.yaml
│
├── docs/                        # Project proposals and reports (PDFs)
├── best.pt                      # Best trained model checkpoint
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/YOLOv7_Gym_Equipment_Detection.git
cd YOLOv7_Gym_Equipment_Detection
```

### 2. Set Up Environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Dataset Setup

The dataset follows the standard YOLO format. Update `dataset/data.yaml` with the absolute paths to your local `images/train`, `images/val`, and `images/test` directories before training.

### 4. Run the Notebooks

Open the notebooks in order:

- **Phase 1**: `phase 1/Project_Update1.ipynb` — Data preparation, augmentation, baseline inference
- **Phase 2**: `Phase 2/Phase2_v2.0.ipynb` — Transfer learning & fine-tuning

---

## 🧠 Model Architecture

- **Base Model**: YOLOv7 (pretrained on COCO)
- **Modifications**: Detection head replaced for 5 custom classes
- **Training Strategy**:
  - Phase 1: Frozen backbone → train detection head only
  - Phase 2: End-to-end fine-tuning (all layers unfrozen)

---

## 📊 Results

| Stage | Description |
|-------|-------------|
| Baseline | Pretrained YOLOv7 inference (no fine-tuning) |
| Phase 2 v1 | Frozen backbone, 5 epochs |
| Phase 2 v2 | Full fine-tuning, best checkpoint saved to `best.pt` |

---

## 📦 Dependencies

See [`requirements.txt`](requirements.txt) for the full list. Key packages:

- `torch` / `torchvision`
- `opencv-python`
- `numpy`, `matplotlib`, `Pillow`
- `PyYAML`, `tqdm`, `scipy`

---

## 📝 Course Information

- **Course**: Deep Learning — University of New Hampshire (UNH)
- **Semester**: Spring 2026

---

## 📄 License

This project is for academic purposes. The YOLOv7 architecture is developed by [WongKinYiu](https://github.com/WongKinYiu/yolov7) and is used under its original license.
