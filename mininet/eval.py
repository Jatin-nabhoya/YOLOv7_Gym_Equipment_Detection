"""
Evaluation utilities for GymDetectorMini.
Matches Part-1 protocol: same NMS settings, torchmetrics MeanAveragePrecision.
"""
import torch
import numpy as np
from .utils import decode_all, nms_single, xywh2xyxy

CLASS_NAMES = ['dumbbell', 'barbell', 'kettlebell', 'resistance_band', 'pull_up_bar']


def evaluate(model, dataloader, anchors_per_scale, strides,
             conf_thres=0.001, iou_thres=0.6, img_size=320, device='cpu'):
    """
    Run full evaluation on a dataloader.

    Returns dict with keys:
        map, map_50, map_75, per_class_ap50  (all floats)
    Matches Part-1 evaluation: conf=0.001 for mAP sweep, iou=0.6 NMS.
    """
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError:
        raise ImportError('pip install torchmetrics>=0.10')

    metric = MeanAveragePrecision(iou_type='bbox', box_format='xyxy',
                                  class_metrics=True)
    metric.to(device)
    model.eval()

    with torch.no_grad():
        for imgs, labels_batch in dataloader:
            imgs     = imgs.to(device)
            raw_list = model(imgs)
            all_pred = decode_all(raw_list, anchors_per_scale, strides)  # (B, N, 5+nc)

            for b in range(len(imgs)):
                pred_b = all_pred[b]
                dets   = nms_single(pred_b, conf_thres=conf_thres, iou_thres=iou_thres)  # (M,6)

                if len(dets):
                    pdict = dict(boxes=dets[:, :4].to(device),
                                 scores=dets[:, 4].to(device),
                                 labels=dets[:, 5].long().to(device))
                else:
                    pdict = dict(boxes=torch.zeros((0,4), device=device),
                                 scores=torch.zeros(0, device=device),
                                 labels=torch.zeros(0, dtype=torch.long, device=device))

                labels = labels_batch[b]
                if isinstance(labels, np.ndarray):
                    labels = torch.from_numpy(labels)
                if len(labels):
                    gt_xywh = labels[:, 1:5].float()
                    gt_xywh[:, 0] *= img_size; gt_xywh[:, 2] *= img_size
                    gt_xywh[:, 1] *= img_size; gt_xywh[:, 3] *= img_size
                    gt_xyxy = xywh2xyxy(gt_xywh)
                    tdict = dict(boxes=gt_xyxy.to(device),
                                 labels=labels[:, 0].long().to(device))
                else:
                    tdict = dict(boxes=torch.zeros((0,4), device=device),
                                 labels=torch.zeros(0, dtype=torch.long, device=device))

                metric.update([pdict], [tdict])

    result = metric.compute()

    per_class_ap50 = {}
    if 'map_per_class' in result and result['map_per_class'].numel() > 0:
        for i, name in enumerate(CLASS_NAMES):
            v = result['map_per_class'][i]
            per_class_ap50[name] = float(v) if not torch.isnan(v) else 0.0

    return {
        'map_50'        : float(result.get('map_50', 0.0)),
        'map'           : float(result.get('map', 0.0)),
        'map_75'        : float(result.get('map_75', 0.0)),
        'per_class_ap50': per_class_ap50,
    }


class EMA:
    """
    Exponential Moving Average of model weights.
    Critical for from-scratch training stability — keeps eval weights smooth.
    """
    def __init__(self, model, decay=0.999):
        self.decay  = decay
        self.shadow = {k: v.data.clone() for k, v in model.named_parameters()}
        self.backup = {}

    def update(self, model):
        d = self.decay
        for k, v in model.named_parameters():
            self.shadow[k] = d * self.shadow[k] + (1 - d) * v.data

    def apply_shadow(self, model):
        for k, v in model.named_parameters():
            self.backup[k] = v.data.clone()
            v.data = self.shadow[k]

    def restore(self, model):
        for k, v in model.named_parameters():
            v.data = self.backup[k]
