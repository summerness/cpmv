from typing import Tuple

import cv2
import numpy as np


def threshold_mask(prob: np.ndarray, threshold: float) -> np.ndarray:
    return (prob > threshold).astype(np.uint8)


def remove_small_regions(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 1
    return cleaned


def postprocess_mask(prob: np.ndarray, threshold: float, min_area: int) -> np.ndarray:
    mask = threshold_mask(prob, threshold)
    mask = remove_small_regions(mask, min_area)
    return mask


def mask_to_prob(mask_logits) -> Tuple[np.ndarray, float]:
    # helper for torch tensors
    if hasattr(mask_logits, "detach"):
        import torch

        prob = torch.sigmoid(mask_logits).detach().cpu().numpy()
        cls_prob = None
    else:
        prob = mask_logits
        cls_prob = None
    return prob, cls_prob
