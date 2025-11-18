from typing import Sequence, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        atrous_rates: Sequence[int],
        out_channels: int = 256,
    ) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )
        for rate in atrous_rates:
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
            if idx == len(self.convs) - 1:
                pooled = conv(x)
                res.append(F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False))
            else:
                res.append(conv(x))
        x = torch.cat(res, dim=1)
        return self.project(x)


class SwinDeepLabV3Plus(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        backbone: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        low_level_idx: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        channels = self.encoder.feature_info.channels()
        aspp_in = channels[-1]
        self.aspp = ASPP(aspp_in, atrous_rates=(6, 12, 18), out_channels=256)
        low_ch = channels[low_level_idx]
        self.low_level = nn.Sequential(
            nn.Conv2d(low_ch, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
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
            nn.Linear(aspp_in, 1),
        )
        self.low_level_idx = low_level_idx

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        high = features[-1]
        low = features[self.low_level_idx]

        aspp_out = self.aspp(high)
        aspp_out = F.interpolate(aspp_out, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low_proj = self.low_level(low)
        decoder_input = torch.cat([aspp_out, low_proj], dim=1)
        decoder_out = self.decoder(decoder_input)
        mask_logits = self.mask_head(decoder_out)
        mask_logits = F.interpolate(mask_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cls_logit = self.cls_head(high)
        return mask_logits, cls_logit.squeeze(-1)
