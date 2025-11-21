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
        identity = x
        if x.ndim == 4 and x.shape[1] < x.shape[-1]:
            x = x.permute(0, 3, 1, 2).contiguous()
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
        patches = patches  # [B, C', K, H, W]
        corr = (att * patches).sum(dim=2)  # [B,C',H,W]

        enhanced = self.proj(corr)

        return identity + enhanced
