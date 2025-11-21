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


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        dims = tuple(range(1, probs.ndim))
        tp = torch.sum(probs * targets, dim=dims)
        fp = torch.sum(probs * (1 - targets), dim=dims)
        fn = torch.sum((1 - probs) * targets, dim=dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, smooth: float = 1e-6) -> None:
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tversky_loss = self.tversky(logits, targets)
        return torch.pow(tversky_loss, 1.0 / self.gamma)


class SegmentationLoss(nn.Module):
    def __init__(self, name: str = "dice", alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75) -> None:
        """
        name: one of ["dice", "tversky", "focal_tversky"]
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        name = name.lower()
        if name == "dice":
            self.aux = DiceLoss()
        elif name == "tversky":
            self.aux = TverskyLoss(alpha=alpha, beta=beta)
        elif name == "focal_tversky":
            self.aux = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        else:
            raise ValueError(f"Unknown seg loss: {name}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets) + self.aux(logits, targets)


class SimilarityConsistencyLoss(nn.Module):
    """
    Encourages high-similarity feature pairs to have consistent mask predictions.
    """

    def __init__(self, topk: int = 8) -> None:
        super().__init__()
        self.topk = topk

    def forward(self, feats: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        feats: [B, C, H, W] feature map
        logits: [B, 1, H, W] mask logits
        """
        b, c, h, w = feats.shape
        n = h * w
        feat_flat = feats.view(b, c, n)
        feat_norm = F.normalize(feat_flat, dim=1)  # [B, C, N]
        sim = torch.bmm(feat_norm.transpose(1, 2), feat_norm)  # [B, N, N]
        k = min(self.topk, n)
        _, idx = torch.topk(sim, k=k, dim=-1)
        # gather logits
        prob = torch.sigmoid(logits)
        prob_flat = prob.view(b, 1, n)  # [B,1,N]
        prob_expand = prob_flat.expand(-1, n, -1)  # [B,N,N]
        gathered = torch.gather(prob_expand, 2, idx)  # [B, N, k]
        # anchor -> [B,N,1] to match gathered last dim
        anchor = prob_flat.transpose(1, 2)  # [B,N,1]
        anchor = anchor.expand(-1, -1, k)
        loss = (anchor - gathered).abs().mean()
        return loss


class MultiTaskLoss(nn.Module):
    """Simple weighted sum without可训练权重."""

    def __init__(self, cls_weight: float = 0.2) -> None:
        super().__init__()
        self.cls_weight = cls_weight

    def forward(self, seg_loss: torch.Tensor, cls_loss: torch.Tensor) -> torch.Tensor:
        return seg_loss + self.cls_weight * cls_loss


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
