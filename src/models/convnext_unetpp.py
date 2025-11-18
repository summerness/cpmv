from typing import Tuple

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class ConvNeXt_UNetPP(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_name: str = "tu-convnext_tiny",
        encoder_weights: str = "imagenet",
        cls_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            decoder_attention_type=None,
        )
        encoder_channels = self.model.encoder.out_channels[-1]
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(cls_dropout),
            nn.Linear(encoder_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.model.encoder(x)
        decoder_output = self.model.decoder(*features)
        mask_logits = self.model.segmentation_head(decoder_output)
        cls_logit = self.cls_head(features[-1])
        return mask_logits, cls_logit.squeeze(-1)
