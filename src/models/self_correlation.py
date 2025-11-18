from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfCorrelationBlock(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4, topk: int = 16) -> None:
        super().__init__()
        reduced_channels = max(in_channels // reduction, 16)
        self.reduce = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.topk = topk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, h, w = x.shape
        feat = self.reduce(x)
        b, rc, _, _ = feat.shape
        feat_flat = feat.view(b, rc, -1)  # [B, C', N]
        norm_feat = F.normalize(feat_flat, dim=1).permute(0, 2, 1)  # [B, N, C']
        sim = torch.bmm(norm_feat, norm_feat.transpose(1, 2))  # [B, N, N]
        topk = min(self.topk, sim.size(-1))
        values, indices = torch.topk(sim, k=topk, dim=-1)
        # Gather neighbor features
        idx_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, norm_feat.size(-1))
        source = norm_feat.unsqueeze(1).expand(-1, norm_feat.size(1), -1, -1)
        gathered = torch.gather(source, 2, idx_expanded)
        corr = gathered.mean(dim=2)  # [B, N, C']
        corr = corr.permute(0, 2, 1).view(b, rc, h, w)
        enhanced = self.proj(corr)
        return identity + enhanced
