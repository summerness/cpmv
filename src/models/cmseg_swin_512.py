from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.self_correlation import SelfCorrelationBlock
from models.utils import safe_timm_create


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CMSegSwin512(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        backbone: str = "swin_tiny_patch4_window7_224",
        enable_aux_corr: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = safe_timm_create(
            backbone,
            pretrained=True,
            in_chans=in_channels,
            features_only=True,
            out_indices=(1, 2, 3),
            img_size=512,
        )
        channels = self.encoder.feature_info.channels()
        self.proj8 = nn.Conv2d(channels[0], 192, kernel_size=1, bias=False)
        self.proj16 = nn.Conv2d(channels[1], 256, kernel_size=1, bias=False)
        self.proj32 = nn.Conv2d(channels[2], 256, kernel_size=1, bias=False)

        self.self_corr = SelfCorrelationBlock(256, reduction=4, window=7)
        self.enable_aux = enable_aux_corr
        if enable_aux_corr:
            self.aux_corr = SelfCorrelationBlock(192, reduction=4, window=5)

        self.up32 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up16 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.dec16 = DecoderBlock(256, 256 + (192 if enable_aux_corr else 0), 256)
        self.dec8 = DecoderBlock(256, 192, 128)
        self.head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor):
        feats = self.encoder(x)
        feat8 = self.proj8(feats[0])
        feat16 = self.proj16(feats[1])
        feat32 = self.proj32(feats[2])

        enhanced16 = self.self_corr(feat16)
        aux_feat = self.aux_corr(feat8) if self.enable_aux else None
        cls_logit = self.cls_head(enhanced16).squeeze(-1)

        up_from32 = self.up32(feat32)
        if aux_feat is not None:
            dec16_input = torch.cat([up_from32, enhanced16, self.up32(aux_feat)], dim=1)
        else:
            dec16_input = torch.cat([up_from32, enhanced16], dim=1)
        dec16 = self.dec16(dec16_input, None)

        up_from16 = self.up16(dec16)
        dec8 = self.dec8(up_from16, feat8)

        mask_logits = self.head(dec8)
        mask_logits = F.interpolate(mask_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return mask_logits, cls_logit, enhanced16
