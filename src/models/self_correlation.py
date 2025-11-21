from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfCorrelationBlock(nn.Module):
    def __init__(self, in_channels, reduction=4, window=7):
        super().__init__()
        reduced = max(in_channels // reduction, 16)
        self.reduce = nn.Conv2d(in_channels, reduced, 1, bias=False)
        self.proj = nn.Conv2d(reduced, in_channels, 1, bias=False)
        self.window = window  # local window

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"SelfCorrelationBlock expects 4D tensor, got {x.shape}")
        # 假定已是 NCHW，调用前若是 NHWC 请先 permute
        identity = x
        b, c, h, w = x.shape

        # reduce dimension
        feat = self.reduce(x)  # [B, C', H, W]

        # local patch unfolding: [B, C', H*W, window*window]
        patches = F.unfold(feat, kernel_size=self.window, padding=self.window // 2)
        patches = patches.view(b, -1, self.window * self.window, h, w)  # [B, C', K, H, W]

        # center vector
        center = feat.unsqueeze(2)  # [B, C', 1, H, W]

        # cosine similarity along patch dimension
        sim = F.cosine_similarity(center, patches, dim=1)  # [B, K, H, W]

        # soft selection of patches (attention)
        att = F.softmax(sim, dim=1).unsqueeze(1)  # [B,1,K,H,W]

        # aggregate patches
        corr = (att * patches).sum(dim=2)  # [B,C',H,W]

        enhanced = self.proj(corr)
        return identity + enhanced


class SelfCorrelationProject(nn.Module):
    """
    自相关特征提取，输出指定通道，不做残差相加。
    用于需要固定输出通道（如 32/512）的场景。
    """

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 4, window: int = 7) -> None:
        super().__init__()
        reduced = max(in_channels // reduction, 16)
        self.reduce = nn.Conv2d(in_channels, reduced, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(reduced, out_channels, kernel_size=1, bias=False)
        self.window = window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"SelfCorrelationProject expects 4D tensor, got {x.shape}")
        b, c, h, w = x.shape
        feat = self.reduce(x)  # [B, C', H, W]
        patches = F.unfold(feat, kernel_size=self.window, padding=self.window // 2)
        patches = patches.view(b, -1, self.window * self.window, h, w)  # [B, C', K, H, W]
        center = feat.unsqueeze(2)  # [B, C', 1, H, W]
        sim = F.cosine_similarity(center, patches, dim=1)  # [B, K, H, W]
        att = F.softmax(sim, dim=1).unsqueeze(1)  # [B,1,K,H,W]
        corr = (att * patches).sum(dim=2)  # [B,C',H,W]
        return self.proj(corr)
