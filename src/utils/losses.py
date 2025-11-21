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
    Copy-Move Consistency Loss: 强制相似 patch 的 mask 预测一致（局部窗口内）。
    feats: [B,C,H,W] (如自相关或 decoder 特征)
    mask_logits: [B,1,H,W] 或其他分辨率（自动对齐）
    """

    def __init__(self, topk: int = 8, patch: int = 7) -> None:
        super().__init__()
        self.topk = topk
        self.patch = patch

    def forward(self, feats: torch.Tensor, mask_logits: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feats.shape
        # 对齐 mask 分辨率
        if mask_logits.shape[-2:] != (h, w):
            mask = F.interpolate(mask_logits, size=(h, w), mode="bilinear", align_corners=False)
        else:
            mask = mask_logits

        # 1) 归一化特征
        f = F.normalize(feats, dim=1)

        # 2) unfold 成局部 patch
        unfold = nn.Unfold(kernel_size=self.patch, padding=self.patch // 2)
        patches = unfold(f)  # [B, C*K*K, H*W]
        patches = patches.view(b, c, self.patch * self.patch, h, w)  # [B,C,K,H,W]

        # 3) 中心特征
        center = f.unsqueeze(2)  # [B,C,1,H,W]

        # 4) 局部 cos 相似度
        sim = F.cosine_similarity(center, patches, dim=1)  # [B,K,H,W]

        # 5) top-k 最相似邻域
        k = min(self.topk, sim.size(1))
        sim_vals, idx = torch.topk(sim, k=k, dim=1)  # idx: [B,k,H,W]

        # 6) gather 邻域 mask
        mask_prob = torch.sigmoid(mask)  # [B,1,H,W]
        mask_neighbors = mask_prob  # [B,1,H,W]
        # 展开便于 gather：先展开为 [B,1,H,W] -> [B,1,1,H,W] 再扩展
        mask_neighbors = mask_prob.unsqueeze(2).expand(-1, 1, k, -1, -1)  # [B,1,k,H,W]
        mask_center = mask_prob.unsqueeze(2)  # [B,1,1,H,W]

        # 7) 特征相似度权重
        w = F.softmax(sim_vals, dim=1).unsqueeze(1)  # [B,1,k,H,W]

        # 8) 预测一致性
        diff = (mask_center - mask_neighbors).abs()
        loss = (w * diff).mean()
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
