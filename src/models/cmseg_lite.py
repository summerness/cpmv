from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import safe_timm_create


class SelfCorrelation(nn.Module):
    """Self-correlation module with optional input concatenation."""

    def __init__(
        self,
        top_k: int = 32,
        downsample: int = 2,
        concat_input: bool = True,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.downsample = downsample
        self.concat_input = concat_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        pooled = F.avg_pool2d(x, kernel_size=self.downsample, stride=self.downsample)
        ph, pw = pooled.shape[-2:]
        features = pooled.view(b, c, -1)
        features = F.normalize(features, dim=1)
        sim = torch.bmm(features.transpose(1, 2), features)
        topk_vals, _ = torch.topk(sim, k=min(self.top_k, sim.size(-1)), dim=-1)
        corr_map = topk_vals.mean(dim=-1).view(b, ph, pw).unsqueeze(1)
        corr_map = F.interpolate(corr_map, size=(h, w), mode="bilinear", align_corners=False)
        if self.concat_input:
            return torch.cat([x, corr_map], dim=1)
        return corr_map


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.skip_channels = skip_channels

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.upsample(x)
        if self.skip_channels and skip is not None:
            x = torch.cat([x, skip], dim=1)
        elif self.skip_channels:
            zeros = x.new_zeros(x.size(0), self.skip_channels, x.size(2), x.size(3))
            x = torch.cat([x, zeros], dim=1)
        return self.conv(x)


class CMSegLite(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        backbone: str = "resnet18",
        pretrained: bool = True,
        multi_scale_corr: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = safe_timm_create(
            backbone,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        channels: Sequence[int] = self.encoder.feature_info.channels()
        encoder_channels = channels[::-1]
        self.multi_scale = multi_scale_corr
        self.primary_corr = SelfCorrelation(top_k=48, concat_input=True)
        self.aux_corr = SelfCorrelation(top_k=24, downsample=1, concat_input=False) if self.multi_scale else None

        corr_channels = encoder_channels[0] + 1 + (1 if self.multi_scale else 0)
        decoder_channels = [256, 128, 64, 32]
        self.decoder_blocks = nn.ModuleList()
        in_ch = corr_channels
        for idx, out_ch in enumerate(decoder_channels):
            skip_ch = encoder_channels[idx + 1] if idx + 1 < len(encoder_channels) else 0
            self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        self.seg_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(corr_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.encoder(x)
        skip_feats = list(reversed(feats[:-1]))
        deep = feats[-1]
        enriched = self.primary_corr(deep)
        if self.multi_scale and self.aux_corr is not None:
            aux = self.aux_corr(feats[-2])
            aux = F.interpolate(aux, size=enriched.shape[-2:], mode="bilinear", align_corners=False)
            enriched = torch.cat([enriched, aux], dim=1)

        cls_logit = self.cls_head(enriched)

        dec = enriched
        for idx, block in enumerate(self.decoder_blocks):
            skip = skip_feats[idx] if idx < len(skip_feats) else None
            dec = block(dec, skip)

        mask_logits = self.seg_head(F.interpolate(dec, size=x.shape[-2:], mode="bilinear", align_corners=False))
        return mask_logits, cls_logit.squeeze(-1)
