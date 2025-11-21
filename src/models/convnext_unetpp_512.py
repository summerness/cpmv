from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import safe_timm_create
from models.self_correlation import SelfCorrelationBlock


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BoundaryRefine(nn.Module):
    """Lightweight boundary refinement to sharpen edges."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.refine(x)


class UNetPPDecoder(nn.Module):
    def __init__(self, encoder_channels: Tuple[int, ...], decoder_channels: Tuple[int, int, int, int]) -> None:
        super().__init__()
        if len(encoder_channels) != 4 or len(decoder_channels) != 4:
            raise ValueError("UNet++ decoder expects four encoder and decoder stages")
        self.channels = decoder_channels
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        ch0, ch1, ch2, ch3 = decoder_channels
        e0, e1, e2, e3 = encoder_channels

        self.x0_0 = ConvBlock(e0, ch0)
        self.x1_0 = ConvBlock(e1, ch1)
        self.x2_0 = ConvBlock(e2, ch2)
        self.x3_0 = ConvBlock(e3, ch3)

        self.x0_1 = ConvBlock(ch0 + ch1, ch0)
        self.x1_1 = ConvBlock(ch1 + ch2, ch1)
        self.x2_1 = ConvBlock(ch2 + ch3, ch2)

        self.x0_2 = ConvBlock(ch0 * 2 + ch1, ch0)
        self.x1_2 = ConvBlock(ch1 * 2 + ch2, ch1)

        self.x0_3 = ConvBlock(ch0 * 3 + ch1, ch0)

    def forward(self, features):
        x0_0 = self.x0_0(features[0])
        x1_0 = self.x1_0(features[1])
        x2_0 = self.x2_0(features[2])
        x3_0 = self.x3_0(features[3])

        x0_1 = self.x0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.x1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x2_1 = self.x2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))

        x0_2 = self.x0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        x1_2 = self.x1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))

        x0_3 = self.x0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        return x0_3


class ConvNeXtUNetPP512(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        backbone: str = "timm/convnext_base.fb_in1k",
        decoder_channels: Tuple[int, int, int, int] = (128, 192, 256, 320),
        cls_dropout: float = 0.2,
        use_self_corr: bool = True,
        self_corr_window: int = 7,
    ) -> None:
        super().__init__()
        self.encoder = safe_timm_create(
            backbone,
            pretrained=True,
            in_chans=in_channels,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        encoder_channels = tuple(self.encoder.feature_info.channels())
        self.use_self_corr = use_self_corr
        if self.use_self_corr:
            # 在 1/16 特征上做自相关增强
            self.self_corr = SelfCorrelationBlock(encoder_channels[2], reduction=4, window=self_corr_window)
            self.corr_proj = nn.Conv2d(encoder_channels[2], decoder_channels[0], kernel_size=1, bias=False)
            self.corr_fuse = nn.Conv2d(decoder_channels[0] * 2, decoder_channels[0], kernel_size=3, padding=1, bias=False)
        self.decoder = UNetPPDecoder(encoder_channels, decoder_channels)
        seg_in_ch = decoder_channels[0]
        self.seg_head = nn.Sequential(
            nn.Conv2d(seg_in_ch, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            BoundaryRefine(64),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(cls_dropout),
            nn.Linear(encoder_channels[-1], 1),
        )

    def forward(self, x: torch.Tensor):
        features = list(self.encoder(x))
        corr_feat = None
        corr_feat_out = None
        if self.use_self_corr:
            features[2] = self.self_corr(features[2])
            corr_feat = self.corr_proj(features[2])
            corr_feat_out = corr_feat

        decoder_out = self.decoder(features)
        if corr_feat is not None:
            corr_up = F.interpolate(corr_feat, size=decoder_out.shape[-2:], mode="bilinear", align_corners=False)
            att = torch.sigmoid(self.corr_fuse(torch.cat([decoder_out, corr_up], dim=1)))
            decoder_out = decoder_out * att + corr_up

        mask_logits = self.seg_head(decoder_out)
        mask_logits = F.interpolate(mask_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cls_logit = self.cls_head(features[-1]).squeeze(-1)
        return mask_logits, cls_logit, corr_feat_out
