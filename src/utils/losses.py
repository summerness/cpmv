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


class MultiTaskLoss(nn.Module):
    """Uncertainty-weighted multi-task loss (Kendall & Gal)."""

    def __init__(self) -> None:
        super().__init__()
        self.log_sigma_seg = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_cls = nn.Parameter(torch.tensor(0.0))

    def forward(self, seg_loss: torch.Tensor, cls_loss: torch.Tensor) -> torch.Tensor:
        sigma_seg = torch.exp(self.log_sigma_seg)
        sigma_cls = torch.exp(self.log_sigma_cls)
        loss = (seg_loss / (2 * sigma_seg**2)) + self.log_sigma_seg
        loss = loss + (cls_loss / (2 * sigma_cls**2)) + self.log_sigma_cls
        return loss


def multitask_loss(
    mask_logits: torch.Tensor,
    mask_targets: torch.Tensor,
    cls_logits: torch.Tensor,
    cls_targets: torch.Tensor,
    cls_weight: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seg_loss_fn = SegmentationLoss()
    seg_loss = seg_loss_fn(mask_logits, mask_targets)
    # 支持单通道 logit，转换为二分类 logits 再做 CE
    if cls_logits.dim() == 1:
        cls_logits = cls_logits.unsqueeze(-1)
    if cls_logits.shape[1] == 1:
        cls_logits = torch.cat([-cls_logits, cls_logits], dim=1)
    elif cls_logits.shape[1] != 2:
        raise ValueError(f"Unexpected cls_logits shape: {cls_logits.shape}")
    cls_loss = F.cross_entropy(cls_logits, cls_targets.long().view(-1))
    total_loss = seg_loss + cls_weight * cls_loss
    return total_loss, seg_loss, cls_loss
