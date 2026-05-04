# CLAUDE.md — Part 2: Mini-Network from Scratch

## What This Project Is

Project Part-2 deliverable for the Deep Learning course (UNH, Spring 2026). Same dataset, same 5 gym equipment classes as Part-1, but instead of fine-tuning a pretrained YOLOv7, we **design a small detection network from scratch and train it end-to-end** with no pretrained weights at all.

Core constraints from the assignment:
1. Mini-network must use **< 10% of YOLOv7 base parameters** → < ~3.7M parameters (YOLOv7 base ≈ 37M)
2. **Train from scratch** — no ImageNet, no COCO, no fine-tuning of any kind
3. **Compare** against the Part-1 fine-tuned YOLOv7 on the same test set
4. Submit **report + code**

This sub-project is built with AI-assistant collaboration (per assignment instructions). Notebook-based, same repo as Part-1.

Team: Varun Gazala, Mohit Raiyani, Jatinkumar Nabhoya.

---

## Repository Layout (additions for Part-2)

```
YOLOv7_Gym_Equipment_Detection/
├── notebooks/
│   ├── 01_phase1_baseline.ipynb        # Part-1 (existing)
│   ├── 02_phase2_finetuning.ipynb      # Part-1 (existing)
│   ├── 03_augmentation_pipeline.ipynb  # Part-1 (existing)
│   ├── 04_gradio_deployment.ipynb      # Part-1 deploy (existing)
│   ├── 05_mininet_train.ipynb          # NEW — Part-2 train mini-net from scratch
│   └── 06_mininet_compare.ipynb        # NEW — Part-2 compare mini vs fine-tuned
├── mininet/                             # NEW — small package for the mini-network
│   ├── __init__.py
│   ├── model.py                         # GymDetectorMini architecture
│   ├── anchors.py                       # k-means anchor generation on dataset
│   ├── loss.py                          # YOLO-style loss (objectness + class + box)
│   ├── dataset.py                       # uses the same dataset/ folder, YOLO labels
│   ├── augment.py                       # light augmentations (re-uses 03 pipeline ideas)
│   ├── utils.py                         # NMS, IoU, decode predictions, plot helpers
│   └── eval.py                          # mAP via torchmetrics, per-class AP
├── weights/
│   ├── best_augmented.pt                # Part-1 fine-tuned YOLOv7
│   └── mininet_best.pt                  # NEW — Part-2 mini-network checkpoint
├── results/
│   ├── augmented_metrics.json           # Part-1 results
│   └── mininet_metrics.json             # NEW — Part-2 results
├── docs/
│   ├── report.pdf                        # Part-1 report (existing)
│   ├── report_part2.pdf / .docx          # NEW — Part-2 comparison report
│   └── figures/
│       ├── mininet_arch.png              # architecture diagram
│       ├── mininet_loss_curves.png       # train/val loss
│       ├── mininet_pr_curve.png
│       └── comparison_*.png              # bar charts mini vs fine-tuned
└── CLAUDE.md                             # Part-1 context (existing)
└── CLAUDE_PART2.md                       # this file
```

---

## Classes (unchanged from Part-1)

Training label order — same as Part-1:

| ID | Name |
|----|------|
| 0 | dumbbell |
| 1 | barbell |
| 2 | kettlebell |
| 3 | resistance_band |
| 4 | pull_up_bar |

`CLASS_NAMES` list in mininet code = source of truth. Do NOT use `dataset/classes.txt` (alphabetical, wrong order).

---

## Parameter Budget

| Model | Params | % of YOLOv7 base |
|-------|--------|------------------|
| YOLOv7 base (Part-1 reference) | ~37.2 M | 100% |
| **Budget for mini-net** | **< 3.72 M** | **< 10%** ✅ |
| Target for our mini-net | ~2.5–3.5 M | ~7–9% |

Mini-net must remain under budget. Always print parameter count at the end of `model.py` definition cell:

```python
n_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {n_params:,}  ({n_params/37_200_000*100:.2f}% of YOLOv7 base)')
assert n_params < 3_720_000, 'Over 10% budget'
```

---

## Mini-Network Architecture (`mininet/model.py`)

`GymDetectorMini` — a small one-stage detector inspired by YOLO, designed to fit in <3.7M params while being trainable from scratch on only 162 (or 648 augmented) images.

**Backbone (~1.2M params)** — 5 stages, each = `Conv-BN-SiLU` + downsample:
- Stem: 3 → 32 channels, stride 2 (320×320)
- Stage 1: 32 → 64, stride 2 (160×160)
- Stage 2: 64 → 128, stride 2 (80×80) ← **P3 output**
- Stage 3: 128 → 192, stride 2 (40×40) ← **P4 output**
- Stage 4: 192 → 256, stride 2 (20×20) ← **P5 output**

Each stage uses a depthwise-separable `MiniBlock` (depthwise 3×3 + pointwise 1×1) with a residual connection, repeated 2× per stage.

**Neck (~0.6M params)** — top-down FPN only (no bottom-up PAN — cheaper):
- P5 → 1×1 conv → upsample → concat with P4 → 3×3 conv
- (P4 fused) → 1×1 conv → upsample → concat with P3 → 3×3 conv
- Outputs: 3 feature maps at strides 8 / 16 / 32

**Head (~0.3M params)** — shared decoupled head, one per scale:
- 3 anchors per scale × (5 box+obj + 5 classes) = `(5+5)*3 = 30` channels
- Same output channel count as Part-1's modified YOLOv7 head — keeps decoding code reusable

**Input size**: 320×320 (smaller than YOLOv7's 640 — quarter the FLOPs, fits in budget). Stride map: 8/16/32 → P3 = 40×40, P4 = 20×20, P5 = 10×10.

**Total target**: ~2.5–3.0M params. Verify with the assertion above.

---

## Anchors (`mininet/anchors.py`)

YOLO-style anchors. Run k-means (k=9) on the **training set ground-truth box dimensions** (normalized) once and save to `mininet/anchors.json`. Then split into 3 groups of 3 by area (small/medium/large) for the 3 scales.

Run once before training:
```python
from mininet.anchors import generate_anchors
anchors = generate_anchors('../dataset/labels/train', n=9, save='mininet/anchors.json')
```

Don't re-run unless dataset changes. Anchors are dataset-specific; YOLOv7 default anchors won't fit our gym equipment box distribution well.

---

## Loss Function (`mininet/loss.py`)

YOLO-v3-style multi-task loss (simpler than YOLOv7's loss for from-scratch training stability):

```
L = λ_box * CIoU_loss(matched_boxes)
  + λ_obj * BCE(objectness, IoU_with_GT)
  + λ_cls * BCE(class_logits, one_hot_GT)
```

**Defaults** (start here, tune if needed):
- `λ_box = 0.05`
- `λ_obj = 1.0` (positive weight = 5.0 because most cells are background)
- `λ_cls = 0.5`

**Anchor matching**: simple — best-IoU anchor per GT box (no SimOTA, no dynamic assignment — those are unstable from scratch on small data).

---

## Training Recipe (`05_mininet_train.ipynb`)

**Dataset**: use **augmented** dataset (648 train images) — small from-scratch nets desperately need data.

**Hyperparameters** (good starting point):
| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-3 with cosine decay to 1e-5 |
| Weight decay | 5e-4 |
| Batch size | 16 (CPU/Mac), 32 (GPU) |
| Epochs | 200 (early stop patience 30 on val mAP) |
| Warmup | linear, first 5 epochs from 1e-5 → 1e-3 |
| Image size | 320×320 |
| EMA | yes, decay 0.999 — critical for from-scratch stability |
| Augmentation | h-flip, color jitter, random affine, mosaic-2 (lightweight) |
| Mixed precision | yes if GPU |

**Why these choices vs Part-1**:
- Lower batch (no pretrained features → more gradient noise tolerated, EMA smooths)
- More epochs (no warm head → needs longer to converge)
- Warmup matters — without it the obj loss diverges in epoch 1
- EMA is non-negotiable from scratch; turning it off makes val mAP very noisy

**Expected training time**: ~45 min on T4 GPU, ~4–6 hours on CPU.

Save `weights/mininet_best.pt` whenever val mAP@0.5 improves.

---

## Evaluation (`mininet/eval.py`)

Same evaluation protocol as Part-1 for a fair comparison:
- Same test set (21 images, 33 boxes)
- Same NMS settings (conf=0.001 for mAP, iou=0.6)
- `torchmetrics.MeanAveragePrecision` — mAP@0.5 and mAP@0.5:0.95
- Per-class AP@0.5
- Confusion matrix at conf=0.25, iou=0.45
- Save to `results/mininet_metrics.json` with the same schema as `results/augmented_metrics.json`

---

## Comparison Notebook (`06_mininet_compare.ipynb`)

Loads both metrics JSON files and produces:

1. **Headline table**:

   | Metric | Mini-Net (scratch) | YOLOv7 fine-tuned (Part-1) | Δ |
   |--------|---|---|---|
   | Params | ~3.0M | 37.2M | mini = 8% |
   | mAP@0.5 | TBD | 0.4603 | TBD |
   | mAP@0.5:0.95 | TBD | 0.2392 | TBD |
   | Inference time / img (CPU) | TBD | TBD | TBD |
   | Model size on disk | ~12 MB | 146 MB | TBD |

2. **Per-class AP bar chart** — side-by-side bars, 5 classes
3. **Side-by-side prediction visualizations** — same 6 test images shown for both models, GT in green, predictions in red
4. **Loss curves** — train vs val for the mini-net only (for the report)
5. **PR curves** — at IoU=0.5, all classes, both models on same axes

---

## Discussion Points for the Report

The expected result: **mini-net will underperform substantially** on mAP. That's the whole point of the comparison. Things the report should cover honestly:

1. **Capacity** — 10× fewer parameters means 10× less representational power; expect mAP drop of 0.15–0.30.
2. **Pretraining matters more than capacity on small datasets**. YOLOv7's COCO weights had seen millions of objects. Our mini-net sees 648 training images, ever.
3. **Inductive bias gap** — YOLOv7's PAN, multi-scale training, SimOTA, advanced loss all help. Mini-net deliberately strips these for stability.
4. **Where mini-net might win**:
   - Inference speed (smaller, less compute)
   - Disk footprint (12 MB vs 146 MB)
   - Edge-device deployability
5. **Where it definitely loses**:
   - Per-class AP, especially on hard classes (barbell, resistance_band)
   - Generalization to unseen poses/lighting

---

## What NOT to Do

- Don't load any pretrained weights, ImageNet stats normalization is fine but **no pretrained convs**. The whole point is from-scratch.
- Don't change the test set or evaluation protocol — must match Part-1 for the comparison to be valid.
- Don't exceed 3.7M parameters. Always assert at model construction.
- Don't use YOLOv7's loss (`ComputeLossOTA`) — it depends on pretrained features stabilizing the SimOTA assignment. Use the simpler v3-style loss above.
- Don't train at 640×640 — quadruples FLOPs and hurts the parameter/compute story without helping mAP much on our small dataset. 320×320 is right.
- Don't compare on the val set — only on the test set, and only after picking the best checkpoint by val mAP.

---

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Loss diverges epoch 1 | No warmup, or LR too high | Use 5-epoch linear warmup, start LR 1e-5 |
| mAP stuck at 0 after 30 epochs | Anchors not matched to dataset | Re-run k-means anchor generation |
| Val mAP very noisy epoch-to-epoch | No EMA | Enable EMA decay 0.999 |
| Loss decreases but mAP stays low | Class loss dominates, model doesn't learn boxes | Lower λ_cls to 0.25, raise λ_box to 0.1 |
| Param count over budget | Too many channels in backbone | Drop stage 4 channels 256 → 192, recheck |
| OOM on CPU/Mac | Batch too big | Drop batch size to 8 |
| `mininet_best.pt` not saved | Val mAP never improved | Check label format — YOLO normalized cx,cy,w,h |

---

## Suggested Run Order

1. Open `notebooks/05_mininet_train.ipynb`
2. Run anchor generation cell once (saves `mininet/anchors.json`)
3. Run training cell — uses augmented dataset by default
4. Best checkpoint auto-saved to `weights/mininet_best.pt`
5. Run eval cell — saves `results/mininet_metrics.json`
6. Open `notebooks/06_mininet_compare.ipynb` — produces all comparison figures + tables
7. Use those figures + tables in `docs/report_part2.docx`

---

## Code Style (matches Part-1)

- Short, focused cells (per Jatin's preference from Part-1)
- Inline `plt.show()`, no `savefig()` unless explicitly making a figure for the report
- Functions ~20–25 lines, clear step-by-step section headings
- One config cell at the top of each notebook with all hyperparameters
- All paths relative to `notebooks/` (e.g. `'../dataset_augmented'`)
