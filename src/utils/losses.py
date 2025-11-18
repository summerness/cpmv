from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        intersection = torch.sum(probs * targets, dim=dims)
        cardinality = torch.sum(probs + targets, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets) + self.dice(logits, targets)


def multitask_loss(
    mask_logits: torch.Tensor,
    mask_targets: torch.Tensor,
    cls_logits: torch.Tensor,
    cls_targets: torch.Tensor,
    cls_weight: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seg_loss_fn = SegmentationLoss()
    seg_loss = seg_loss_fn(mask_logits, mask_targets)
    cls_loss = F.binary_cross_entropy_with_logits(cls_logits.view(-1), cls_targets)
    total_loss = seg_loss + cls_weight * cls_loss
    return total_loss, seg_loss, cls_loss
