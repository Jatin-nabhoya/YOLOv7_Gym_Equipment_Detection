"""
Light augmentation pipeline for training GymDetectorMini from scratch.
All transforms operate on (img: HWC RGB uint8, labels: (N,5) float32 numpy).
Labels format: (cls, cx, cy, w, h) normalized to [0,1].
"""
import random
import cv2
import numpy as np


# ── Primitive transforms ───────────────────────────────────────────────────────

def h_flip(img, labels):
    img = img[:, ::-1, :].copy()
    if len(labels):
        lbl = labels.copy()
        lbl[:, 1] = 1.0 - labels[:, 1]   # cx mirror
        labels = lbl
    return img, labels


def color_jitter(img, brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05):
    img = img.astype(np.float32) / 255.0
    # brightness
    img = np.clip(img * (1 + random.uniform(-brightness, brightness)), 0, 1)
    # contrast
    gray = img.mean()
    img  = np.clip(img * (1 + random.uniform(-contrast, contrast)) +
                   gray * random.uniform(-0.1, 0.1), 0, 1)
    img_u8 = (img * 255).astype(np.uint8)

    # saturation + hue in HSV
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-hue * 180, hue * 180)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + random.uniform(-saturation, saturation)), 0, 255)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    return img, labels


def random_affine(img, labels, degrees=5, translate=0.05, scale=(0.9, 1.1)):
    h, w = img.shape[:2]
    angle    = random.uniform(-degrees, degrees)
    tx       = random.uniform(-translate, translate) * w
    ty       = random.uniform(-translate, translate) * h
    sf       = random.uniform(*scale)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, sf)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))

    if len(labels):
        lbl = labels.copy()
        # Transform box centers; keep w/h unchanged (close to true for small transforms)
        cx = lbl[:, 1] * w
        cy = lbl[:, 2] * h
        pts = np.stack([cx, cy, np.ones(len(lbl))], axis=1)   # (N, 3)
        new_pts = (M @ pts.T).T                                 # (N, 2)
        lbl[:, 1] = np.clip(new_pts[:, 0] / w, 0.01, 0.99)
        lbl[:, 2] = np.clip(new_pts[:, 1] / h, 0.01, 0.99)
        labels = lbl
    return img, labels


# ── Composed pipeline ─────────────────────────────────────────────────────────

class TrainAugment:
    """
    Standard training augmentation for from-scratch detection.
    Deliberately light — heavy augmentation is already handled by dataset_augmented.
    """
    def __init__(self, p_flip=0.5, p_jitter=0.5, p_affine=0.3):
        self.p_flip   = p_flip
        self.p_jitter = p_jitter
        self.p_affine = p_affine

    def __call__(self, img, labels):
        if random.random() < self.p_flip:
            img, labels = h_flip(img, labels)
        if random.random() < self.p_jitter:
            img, labels = color_jitter(img, labels)
        if random.random() < self.p_affine:
            img, labels = random_affine(img, labels)
        return img, labels
