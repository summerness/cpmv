from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import safe_timm_create
from models.self_correlation import SelfCorrelationBlock


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256, atrous_rates: Sequence[int] = (1, 6, 12, 18)) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )
        for rate in atrous_rates[1:]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(modules), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for idx, conv in enumerate(self.convs):
            feat = conv(x)
            if idx == len(self.convs) - 1:
                feat = F.interpolate(feat, size=x.shape[-2:], mode="bilinear", align_corners=False)
            res.append(feat)
        x = torch.cat(res, dim=1)
        return self.project(x)


class SwinDeepLab512(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        backbone: str = "timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        low_level_idx: int = 0,
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
            img_size=512,
        )
        channels = self.encoder.feature_info.channels()
        self.aspp = ASPP(channels[-1], out_channels=256)
        self.low_proj = nn.Sequential(
            nn.Conv2d(channels[low_level_idx], 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.use_self_corr = use_self_corr
        if self.use_self_corr:
            # 在较深层特征上做自相关（取倒数第二层，通常 1/16）
            self.self_corr = SelfCorrelationBlock(channels[-2], reduction=4, window=self_corr_window)
            self.corr_proj = nn.Conv2d(channels[-2], 256, kernel_size=1, bias=False)
            self.corr_fuse = nn.Conv2d(256 + 64, 256, kernel_size=1, bias=False)
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 64, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.mask_head = nn.Conv2d(256, num_classes, kernel_size=1)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
        )
        self.low_level_idx = low_level_idx

    def forward(self, x: torch.Tensor):
        features = self.encoder(x)
        high = features[-1]
        low = features[self.low_level_idx]
        corr_feat = None
        if self.use_self_corr:
            mid = features[-2]
            # 若为通道末尾格式，转换为 NCHW
            if mid.ndim == 4 and mid.shape[1] < mid.shape[-1]:
                mid = mid.permute(0, 3, 1, 2).contiguous()
            mid = self.self_corr(mid)
            corr_feat = self.corr_proj(mid)

        aspp_out = self.aspp(high)
        cls_logit = self.cls_head(aspp_out).squeeze(-1)
        upsampled = F.interpolate(aspp_out, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low_feat = self.low_proj(low)
        decoder_in = torch.cat([upsampled, low_feat], dim=1)
        if corr_feat is not None:
            corr_up = F.interpolate(corr_feat, size=low.shape[-2:], mode="bilinear", align_corners=False)
            att = torch.sigmoid(self.corr_fuse(torch.cat([upsampled, low_feat], dim=1)))
            decoder_in = decoder_in * (1 + att)
        decoder_out = self.decoder(decoder_in)
        mask_logits = self.mask_head(decoder_out)
        mask_logits = F.interpolate(mask_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return mask_logits, cls_logit, corr_feat
