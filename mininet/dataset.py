import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class GymMiniDataset(Dataset):
    """
    YOLO-format dataset.
    Labels: class cx cy w h  (normalized to original image size)
    Returns:
        img_t  : (3, img_size, img_size) float32 tensor in [0, 1], RGB
        labels : (N, 5) float32 numpy array — (cls, cx, cy, w, h) normalized
                 to the letterboxed image. Empty → shape (0, 5).
    """
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

    def __init__(self, img_dir, label_dir, img_size=320, transform=None):
        self.img_size  = img_size
        self.transform = transform
        self.label_dir = Path(label_dir)
        self.img_paths = sorted(
            p for p in Path(img_dir).iterdir()
            if p.suffix.lower() in self.IMG_EXTS
        )
        assert len(self.img_paths), f'No images found in {img_dir}'

    def __len__(self):
        return len(self.img_paths)

    def _load_label(self, stem):
        p = self.label_dir / (stem + '.txt')
        if not p.exists():
            return np.zeros((0, 5), dtype=np.float32)
        lines = [l.strip() for l in p.read_text().strip().splitlines() if l.strip()]
        if not lines:
            return np.zeros((0, 5), dtype=np.float32)
        return np.array([list(map(float, l.split())) for l in lines], dtype=np.float32)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        labels = self._load_label(img_path.stem)   # (N, 5) normalized

        if self.transform is not None:
            img, labels = self.transform(img, labels)

        orig_h, orig_w = img.shape[:2]
        img, ratio, (pad_l, pad_t) = _letterbox(img, self.img_size)

        # Adjust labels from original-image-normalized → letterboxed-image-normalized
        if len(labels):
            lbl = labels.copy()
            new_w = orig_w * ratio
            new_h = orig_h * ratio
            lbl[:, 1] = (labels[:, 1] * new_w + pad_l) / self.img_size   # cx
            lbl[:, 2] = (labels[:, 2] * new_h + pad_t) / self.img_size   # cy
            lbl[:, 3] =  labels[:, 3] * new_w           / self.img_size   # w
            lbl[:, 4] =  labels[:, 4] * new_h           / self.img_size   # h
            labels = lbl

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_t, labels


def _letterbox(img, size):
    """
    Resize + pad to size×size preserving aspect ratio (grey pad = 114).
    Returns (img, ratio, (pad_left, pad_top)).
    """
    h, w = img.shape[:2]
    ratio = size / max(h, w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_l = (size - new_w) // 2
    pad_r = size - new_w - pad_l
    pad_t = (size - new_h) // 2
    pad_b = size - new_h - pad_t
    img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, ratio, (pad_l, pad_t)


def collate_fn(batch):
    """Stack images; keep labels as a list (variable N per image)."""
    imgs, labels = zip(*batch)
    return torch.stack(imgs), list(labels)
