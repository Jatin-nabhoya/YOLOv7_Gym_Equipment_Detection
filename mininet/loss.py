"""
YOLOv3-style multi-task loss for GymDetectorMini.

L = λ_box * CIoU  +  λ_obj * BCE(obj)  +  λ_cls * BCE(cls)

Anchor matching: best-IoU anchor per GT box (simple, stable from scratch).
"""
import torch
import torch.nn as nn
import numpy as np


# ── CIoU loss ─────────────────────────────────────────────────────────────────

def ciou_loss(pred, target, eps=1e-7):
    """
    Differentiable CIoU between (N,4) tensors in (cx,cy,w,h) pixel coords.
    Returns per-element loss tensor of shape (N,).
    """
    px1 = pred[:, 0] - pred[:, 2] / 2;  py1 = pred[:, 1] - pred[:, 3] / 2
    px2 = pred[:, 0] + pred[:, 2] / 2;  py2 = pred[:, 1] + pred[:, 3] / 2
    tx1 = target[:, 0] - target[:, 2] / 2; ty1 = target[:, 1] - target[:, 3] / 2
    tx2 = target[:, 0] + target[:, 2] / 2; ty2 = target[:, 1] + target[:, 3] / 2

    inter = (torch.minimum(px2, tx2) - torch.maximum(px1, tx1)).clamp(0) * \
            (torch.minimum(py2, ty2) - torch.maximum(py1, ty1)).clamp(0)
    union = (px2-px1)*(py2-py1) + (tx2-tx1)*(ty2-ty1) - inter + eps
    iou   = inter / union

    enc_x1 = torch.minimum(px1, tx1); enc_y1 = torch.minimum(py1, ty1)
    enc_x2 = torch.maximum(px2, tx2); enc_y2 = torch.maximum(py2, ty2)
    c2  = (enc_x2-enc_x1)**2 + (enc_y2-enc_y1)**2 + eps
    d2  = (pred[:, 0]-target[:, 0])**2 + (pred[:, 1]-target[:, 1])**2

    v     = (4 / (torch.pi**2)) * (torch.atan(target[:, 2]/(target[:, 3]+eps)) -
                                    torch.atan(pred[:, 2]/(pred[:, 3]+eps)))**2
    alpha = v / (1 - iou + v + eps)
    return 1 - iou + d2/c2 + alpha*v


# ── Main loss class ────────────────────────────────────────────────────────────

class YOLOLoss(nn.Module):
    """
    anchors_per_scale : [[(w,h)×3], [(w,h)×3], [(w,h)×3]]  — pixel coords
    strides           : [8, 16, 32]
    img_size          : 320
    """
    def __init__(self, anchors_per_scale, strides, nc=5, img_size=320,
                 lambda_box=0.05, lambda_obj=1.0, lambda_cls=0.5, obj_pos_w=5.0):
        super().__init__()
        self.anchors   = anchors_per_scale
        self.strides   = strides
        self.nc        = nc
        self.img_size  = img_size
        self.lbox      = lambda_box
        self.lobj      = lambda_obj
        self.lcls      = lambda_cls
        self.bce_obj   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([obj_pos_w]))
        self.bce_cls   = nn.BCEWithLogitsLoss()

    def forward(self, raw_list, targets_batch):
        """
        raw_list     : [P3, P4, P5]  each (B, A*(5+nc), H, W) raw logits
        targets_batch: list of B arrays (N_i, 5) = cls,cx,cy,w,h normalized
        Returns (scalar_loss, dict_of_components).
        """
        device = raw_list[0].device
        self.bce_obj.pos_weight = self.bce_obj.pos_weight.to(device)

        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)
        loss_cls = torch.zeros(1, device=device)

        for raw, anchors, stride in zip(raw_list, self.anchors, self.strides):
            B, _, H, W = raw.shape
            A  = len(anchors)
            nc = self.nc

            # (B, A, H, W, 5+nc)
            pred = raw.view(B, A, 5+nc, H, W).permute(0, 1, 3, 4, 2).contiguous()

            t_obj     = torch.zeros(B, A, H, W, device=device)
            pred_boxes, targ_boxes = [], []
            pred_cls,  targ_cls    = [], []

            for b, labels in enumerate(targets_batch):
                if labels is None or (hasattr(labels, '__len__') and len(labels) == 0):
                    continue
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()
                if not isinstance(labels, np.ndarray):
                    labels = np.array(labels, dtype=np.float32)
                if labels.ndim == 1:
                    labels = labels[None]

                for row in labels:
                    cls, cx, cy, w, h = row
                    # pixel coords in letterboxed image
                    cx_px = cx * self.img_size
                    cy_px = cy * self.img_size
                    w_px  = w  * self.img_size
                    h_px  = h  * self.img_size

                    # Pick best anchor by IoU (size only)
                    best_iou, best_a = -1.0, 0
                    for ai, (aw, ah) in enumerate(anchors):
                        iou_ = (min(w_px,aw)*min(h_px,ah)) / \
                               (w_px*h_px + aw*ah - min(w_px,aw)*min(h_px,ah) + 1e-9)
                        if iou_ > best_iou:
                            best_iou, best_a = iou_, ai

                    gi = int(min(cx * W, W - 1))
                    gj = int(min(cy * H, H - 1))

                    t_obj[b, best_a, gj, gi] = 1.0

                    aw, ah = anchors[best_a]
                    p_raw  = pred[b, best_a, gj, gi]       # (5+nc,) with grad

                    pcx = (torch.sigmoid(p_raw[0]) + gi) * stride
                    pcy = (torch.sigmoid(p_raw[1]) + gj) * stride
                    pw  = torch.exp(p_raw[2].clamp(-4, 4)) * aw
                    ph  = torch.exp(p_raw[3].clamp(-4, 4)) * ah

                    pred_boxes.append(torch.stack([pcx, pcy, pw, ph]))
                    targ_boxes.append(
                        torch.tensor([cx_px, cy_px, w_px, h_px], device=device, dtype=torch.float32)
                    )

                    t_cls_vec = torch.zeros(nc, device=device)
                    t_cls_vec[int(cls)] = 1.0
                    targ_cls.append(t_cls_vec)
                    pred_cls.append(p_raw[5:])

            loss_obj = loss_obj + self.bce_obj(pred[..., 4], t_obj)

            if pred_boxes:
                loss_box = loss_box + ciou_loss(
                    torch.stack(pred_boxes), torch.stack(targ_boxes)
                ).mean()
                loss_cls = loss_cls + self.bce_cls(
                    torch.stack(pred_cls), torch.stack(targ_cls)
                )

        total = self.lbox*loss_box + self.lobj*loss_obj + self.lcls*loss_cls
        return total.squeeze(), {
            'box': loss_box.item(),
            'obj': loss_obj.item(),
            'cls': loss_cls.item(),
        }
