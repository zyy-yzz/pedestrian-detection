"""
WIoU Loss: Wise-IoU attention-based box regression loss.

Focuses training on hard examples by down-weighting easy boxes
(high IoU) and amplifying loss for low-IoU predictions.
Replaces CIoU in YOLOv8 detection head.
"""

import torch
import torch.nn as nn


def wiou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    delta: float = 1.9,
) -> torch.Tensor:
    """
    Wise-IoU v1 loss with attention-based weighting.

    Args:
        pred_boxes:  [N, 4]  predicted boxes (xyxy or cxcywh in normalised coords)
        target_boxes: [N, 4]  ground-truth boxes
        delta: focality hyper-parameter controlling hard/easy weighting

    Returns:
        scalar loss (mean over batch)

    Formula:
        IoU = intersection / union
        beta = (1 - IoU) / delta
        R_wiou = exp(beta)
        L_wiou = mean(R_wiou * (1 - IoU))

    High IoU → small beta → small weight (easy example)
    Low IoU  → large beta → large weight (hard example)
    """

    # Convert to xyxy if needed (assume cxcywh input)
    if pred_boxes.shape[-1] == 4:
        pred_xyxy = _cxcywh_to_xyxy(pred_boxes)
        target_xyxy = _cxcywh_to_xyxy(target_boxes)
    else:
        pred_xyxy = pred_boxes
        target_xyxy = target_boxes

    iou = _box_iou(pred_xyxy, target_xyxy)  # [N]
    iou = iou.clamp(min=1e-7)

    # Wise weighting
    beta = (1.0 - iou) / delta
    wise_weight = torch.exp(beta)

    loss = (wise_weight * (1.0 - iou)).mean()
    return loss


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert from [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes (both xyxy)."""
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1 + area2 - inter + 1e-7
    return inter / union


class WIoULoss(nn.Module):
    """
    WIoU loss as a nn.Module for integration into training loops.
    """

    def __init__(self, delta: float = 1.9):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return wiou_loss(pred, target, self.delta)
