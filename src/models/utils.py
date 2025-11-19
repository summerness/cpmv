import logging

import timm


def safe_timm_create(backbone: str, *, pretrained: bool = True, **kwargs):
    """Create timm model with graceful fallback when pretrained weights download fails."""

    try:
        return timm.create_model(backbone, pretrained=pretrained, **kwargs)
    except Exception as exc:  # pragma: no cover - network/download errors
        logging.warning(
            "Failed to load pretrained weights for %s due to %s; falling back to random init.",
            backbone,
            exc,
        )
        return timm.create_model(backbone, pretrained=False, **kwargs)
