"""
K-means anchor generation on training set GT boxes.
Run once; saves anchors.json to mininet/ directory.
"""
import json
import numpy as np
from pathlib import Path


def _load_gt_wh(labels_dir):
    """Return (N, 2) array of (w, h) normalized box dims from YOLO .txt files."""
    wh = []
    for p in Path(labels_dir).glob('*.txt'):
        for line in p.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                wh.append([float(parts[3]), float(parts[4])])
    return np.array(wh, dtype=np.float32)


def _kmeans_iou(wh, k=9, max_iters=300, seed=42):
    """
    K-means on box dimensions using 1-IoU as distance.
    Assumes center-aligned boxes (only size matters for matching).
    """
    rng = np.random.default_rng(seed)
    n = len(wh)
    centroids = wh[rng.choice(n, k, replace=False)].copy()

    for _ in range(max_iters):
        # IoU between every box and every centroid (center-aligned)
        min_wh = np.minimum(wh[:, None, :], centroids[None, :, :])  # (n, k, 2)
        inter  = min_wh[..., 0] * min_wh[..., 1]                    # (n, k)
        box_a  = wh[:, 0] * wh[:, 1]
        cen_a  = centroids[:, 0] * centroids[:, 1]
        union  = box_a[:, None] + cen_a[None, :] - inter
        iou    = inter / (union + 1e-9)
        assign = iou.argmax(axis=1)

        new_c = np.stack([
            wh[assign == i].mean(axis=0) if (assign == i).any() else centroids[i]
            for i in range(k)
        ])
        if np.allclose(centroids, new_c, atol=1e-6):
            break
        centroids = new_c

    return centroids


def generate_anchors(labels_dir, n=9, img_size=320, save=None):
    """
    Generate n anchors via k-means, sort by area, split into 3 FPN groups.

    Returns list of 3 groups (small / medium / large):
        [ [(w,h),(w,h),(w,h)],  # stride-8  (P3)
          [(w,h),(w,h),(w,h)],  # stride-16 (P4)
          [(w,h),(w,h),(w,h)] ] # stride-32 (P5)
    All (w, h) values are in pixels for img_size×img_size input.
    """
    wh = _load_gt_wh(labels_dir)
    assert len(wh) >= n, f'Only {len(wh)} boxes found in {labels_dir}, need >= {n}'

    centroids_norm = _kmeans_iou(wh, k=n)
    centroids_px   = centroids_norm * img_size

    order      = np.argsort(centroids_px[:, 0] * centroids_px[:, 1])
    sorted_px  = centroids_px[order]

    groups = [
        [sorted_px[i].tolist() for i in range(0, 3)],   # smallest  → P3
        [sorted_px[i].tolist() for i in range(3, 6)],   # medium    → P4
        [sorted_px[i].tolist() for i in range(6, 9)],   # largest   → P5
    ]

    if save:
        out = {'anchors': groups, 'img_size': img_size, 'n_boxes': int(len(wh))}
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        Path(save).write_text(json.dumps(out, indent=2))
        print(f'Anchors saved → {save}')
        for si, (g, stride) in enumerate(zip(groups, [8, 16, 32])):
            print(f'  P{3+si} (stride {stride}): ' +
                  ', '.join(f'({w:.1f}×{h:.1f})' for w, h in g))

    return groups


def load_anchors(path):
    """Load anchors from JSON. Returns list of 3 groups of (w, h) pixel tuples."""
    data = json.loads(Path(path).read_text())
    return data['anchors']
