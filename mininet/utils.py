"""
Inference utilities: decode raw predictions, NMS, IoU helpers, plot helpers.
"""
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CLASS_NAMES = ['dumbbell', 'barbell', 'kettlebell', 'resistance_band', 'pull_up_bar']
COLORS_HEX  = ['#e63946', '#2a9d8f', '#457b9d', '#e9c46a', '#8338ec']
# OpenCV BGR equivalents (for cv2 drawing)
COLORS_BGR = [(50,57,230), (143,157,42), (155,123,69), (106,196,233), (236,56,131)]


# ── Box conversion ─────────────────────────────────────────────────────────────

def xywh2xyxy(t):
    """(cx,cy,w,h) → (x1,y1,x2,y2). Works on tensors or numpy arrays."""
    x1 = t[..., 0] - t[..., 2] / 2
    y1 = t[..., 1] - t[..., 3] / 2
    x2 = t[..., 0] + t[..., 2] / 2
    y2 = t[..., 1] + t[..., 3] / 2
    if isinstance(t, torch.Tensor):
        return torch.stack([x1, y1, x2, y2], dim=-1)
    return np.stack([x1, y1, x2, y2], axis=-1)


# ── Decode ─────────────────────────────────────────────────────────────────────

def decode_predictions(raw, anchors, stride):
    """
    Decode one scale's raw output.

    raw     : (B, A*(5+nc), H, W)
    anchors : [(w,h), ...] A entries, pixel coords
    stride  : int

    Returns (B, A*H*W, 5+nc) with decoded (cx,cy,w,h,obj,cls...) in pixels.
    """
    B, _, H, W = raw.shape
    A  = len(anchors)
    nc = raw.shape[1] // A - 5

    pred = raw.view(B, A, 5+nc, H, W).permute(0, 1, 3, 4, 2).contiguous()  # (B,A,H,W,5+nc)

    gy = torch.arange(H, device=raw.device, dtype=torch.float32).view(1, 1, H, 1)
    gx = torch.arange(W, device=raw.device, dtype=torch.float32).view(1, 1, 1, W)
    aw = torch.tensor([a[0] for a in anchors], device=raw.device, dtype=torch.float32).view(1, A, 1, 1)
    ah = torch.tensor([a[1] for a in anchors], device=raw.device, dtype=torch.float32).view(1, A, 1, 1)

    cx  = ((torch.sigmoid(pred[..., 0]) + gx) * stride).unsqueeze(-1)
    cy  = ((torch.sigmoid(pred[..., 1]) + gy) * stride).unsqueeze(-1)
    pw  = (torch.exp(pred[..., 2].clamp(-4, 4)) * aw).unsqueeze(-1)
    ph  = (torch.exp(pred[..., 3].clamp(-4, 4)) * ah).unsqueeze(-1)
    obj = torch.sigmoid(pred[..., 4:5])
    cls = torch.sigmoid(pred[..., 5:])

    decoded = torch.cat([cx, cy, pw, ph, obj, cls], dim=-1)  # (B,A,H,W,5+nc)
    return decoded.view(B, A*H*W, 5+nc)


def decode_all(raw_list, anchors_per_scale, strides):
    """Decode all 3 scales and concat → (B, total_preds, 5+nc)."""
    return torch.cat([
        decode_predictions(raw, anc, s)
        for raw, anc, s in zip(raw_list, anchors_per_scale, strides)
    ], dim=1)


# ── NMS ────────────────────────────────────────────────────────────────────────

def nms_single(pred, conf_thres=0.25, iou_thres=0.45):
    """
    pred : (N, 5+nc) decoded — (cx,cy,w,h,obj,cls...)
    Returns (M, 6) tensor: (x1,y1,x2,y2, conf, cls_id)  or shape (0,6).
    """
    obj       = pred[:, 4]
    cls_conf, cls_id = pred[:, 5:].max(dim=1)
    conf      = obj * cls_conf

    mask = conf > conf_thres
    pred    = pred[mask];  conf = conf[mask];  cls_id = cls_id[mask]

    if len(pred) == 0:
        return torch.zeros((0, 6), device=pred.device)

    boxes_xyxy = xywh2xyxy(pred[:, :4])
    keep_rows  = []

    for c in cls_id.unique():
        m  = (cls_id == c)
        b  = boxes_xyxy[m]
        s  = conf[m]
        ki = _greedy_nms(b, s, iou_thres)
        idx = torch.where(m)[0][ki]
        for i in idx:
            keep_rows.append(torch.cat([boxes_xyxy[i], conf[i:i+1], cls_id[i:i+1].float()]))

    return torch.stack(keep_rows) if keep_rows else torch.zeros((0, 6), device=pred.device)


def _greedy_nms(boxes, scores, iou_thres):
    """Standard greedy NMS. Returns list of kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = (x2-x1)*(y2-y1)
    order  = scores.argsort(descending=True)
    keep   = []
    while len(order):
        i = order[0].item(); keep.append(i)
        if len(order) == 1: break
        rest = order[1:]
        xx1  = x1[rest].clamp(min=x1[i].item()); yy1 = y1[rest].clamp(min=y1[i].item())
        xx2  = x2[rest].clamp(max=x2[i].item()); yy2 = y2[rest].clamp(max=y2[i].item())
        inter= ((xx2-xx1).clamp(0)) * ((yy2-yy1).clamp(0))
        iou  = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order= rest[iou <= iou_thres]
    return keep


# ── Plotting ───────────────────────────────────────────────────────────────────

def draw_boxes_cv2(img_rgb, dets, line=2, font_scale=0.5):
    """
    Draw bounding boxes on a copy of img_rgb (HWC uint8).
    dets: (M,6) tensor (x1,y1,x2,y2,conf,cls_id) or list of (name,conf,x1,y1,x2,y2).
    Returns annotated HWC RGB array.
    """
    out = img_rgb.copy()
    if isinstance(dets, torch.Tensor):
        dets = dets.cpu().numpy()
        rows = [(CLASS_NAMES[int(d[5])], d[4], *d[:4].astype(int)) for d in dets]
    else:
        rows = dets

    for name, conf, x1, y1, x2, y2 in rows:
        ci    = CLASS_NAMES.index(name) if name in CLASS_NAMES else 0
        color = COLORS_BGR[ci]
        label = f'{name} {conf:.2f}'
        cv2.rectangle(out, (x1, y1), (x2, y2), color, line)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(out, (x1, max(0, y1-th-6)), (x1+tw+4, y1), color, -1)
        cv2.putText(out, label, (x1+2, max(4, y1-3)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 1, cv2.LINE_AA)
    return out


def plot_detections(img_rgb, dets, title='', figsize=(7, 7)):
    """Matplotlib plot of bounding boxes. dets same format as draw_boxes_cv2."""
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(draw_boxes_cv2(img_rgb, dets))
    ax.set_title(title, fontsize=11)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
