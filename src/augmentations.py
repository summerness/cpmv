import random
from typing import Dict, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


def _add_flip(transforms, prob: float, vertical: bool = False) -> None:
    if prob and prob > 0:
        aug = A.VerticalFlip if vertical else A.HorizontalFlip
        transforms.append(aug(p=prob))


def get_train_augmentations(image_size: Tuple[int, int], params: Optional[Dict] = None) -> A.Compose:
    cfg = params or {}
    h, w = image_size
    transforms = [A.Resize(height=h, width=w)]

    _add_flip(transforms, cfg.get("horizontal_flip", 0.5))
    _add_flip(transforms, cfg.get("vertical_flip", 0.0), vertical=True)

    rotate90_p = cfg.get("random_rotate90", 0.5)
    if rotate90_p and rotate90_p > 0:
        transforms.append(A.RandomRotate90(p=rotate90_p))

    ssr_cfg = cfg.get("shift_scale_rotate", {"p": 0.5, "shift_limit": 0.05, "scale_limit": 0.1, "rotate_limit": 30})
    if ssr_cfg.get("p", 0) > 0:
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=ssr_cfg.get("shift_limit", 0.05),
                scale_limit=ssr_cfg.get("scale_limit", 0.1),
                rotate_limit=ssr_cfg.get("rotate_limit", 30),
                border_mode=cv2.BORDER_REFLECT_101,
                p=ssr_cfg.get("p", 0.5),
            )
        )

    bc_cfg = cfg.get("brightness_contrast", {"p": 0.5, "brightness_limit": 0.2, "contrast_limit": 0.2})
    if bc_cfg.get("p", 0) > 0:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=bc_cfg.get("brightness_limit", 0.2),
                contrast_limit=bc_cfg.get("contrast_limit", 0.2),
                p=bc_cfg.get("p", 0.5),
            )
        )

    blur_cfg = cfg.get("gaussian_blur", {"p": 0.0})
    if blur_cfg.get("p", 0) > 0:
        transforms.append(A.GaussianBlur(blur_limit=blur_cfg.get("blur_limit", 3), p=blur_cfg["p"]))

    noise_cfg = cfg.get("gauss_noise", {"p": 0.0})
    if noise_cfg.get("p", 0) > 0:
        transforms.append(A.GaussNoise(var_limit=tuple(noise_cfg.get("var_limit", (5.0, 20.0))), p=noise_cfg["p"]))

    elastic_cfg = cfg.get("elastic", {})
    if elastic_cfg.get("enabled", False):
        transforms.append(
            A.ElasticTransform(
                p=elastic_cfg.get("p", 0.2),
                alpha=elastic_cfg.get("alpha", 80),
                sigma=elastic_cfg.get("sigma", 12),
                alpha_affine=elastic_cfg.get("alpha_affine", 10),
            )
        )

    grid_cfg = cfg.get("grid_distortion", {})
    if grid_cfg.get("enabled", False):
        transforms.append(A.GridDistortion(p=grid_cfg.get("p", 0.2)))

    normalize_cfg = cfg.get("normalize", {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)})
    transforms.extend(
        [
            A.Normalize(mean=tuple(normalize_cfg.get("mean", (0.5, 0.5, 0.5))), std=tuple(normalize_cfg.get("std", (0.5, 0.5, 0.5)))),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms, additional_targets={"mask": "mask"})


def get_valid_augmentations(image_size: Tuple[int, int], params: Optional[Dict] = None) -> A.Compose:
    cfg = params or {}
    h, w = image_size
    normalize_cfg = cfg.get("normalize", {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)})
    return A.Compose(
        [
            A.Resize(height=h, width=w),
            A.Normalize(mean=tuple(normalize_cfg.get("mean", (0.5, 0.5, 0.5))), std=tuple(normalize_cfg.get("std", (0.5, 0.5, 0.5)))),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )


def synthetic_copy_move(image: np.ndarray, mask: Optional[np.ndarray], p: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """随机制造一块伪造区域并更新 mask."""

    if image.ndim != 3:
        raise ValueError("Expect HWC RGB image for synthetic copy-move")
    if random.random() > p:
        return image, mask if mask is not None else np.zeros(image.shape[:2], dtype=np.uint8)

    h, w, _ = image.shape
    min_h = max(1, int(0.1 * h))
    max_h = max(min_h, int(0.3 * h))
    min_w = max(1, int(0.1 * w))
    max_w = max(min_w, int(0.3 * w))
    patch_h = random.randint(min_h, max_h)
    patch_w = random.randint(min_w, max_w)

    y1 = random.randint(0, h - patch_h)
    x1 = random.randint(0, w - patch_w)
    y2 = y1 + patch_h
    x2 = x1 + patch_w

    patch_img = image[y1:y2, x1:x2].copy()
    # generate irregular shape
    shape = np.zeros((patch_h, patch_w), dtype=np.uint8)
    num_shapes = random.randint(1, 3)
    for _ in range(num_shapes):
        pts = []
        cx = random.randint(patch_w // 4, patch_w * 3 // 4)
        cy = random.randint(patch_h // 4, patch_h * 3 // 4)
        for _ in range(random.randint(3, 6)):
            dx = random.randint(-patch_w // 4, patch_w // 4)
            dy = random.randint(-patch_h // 4, patch_h // 4)
            pts.append([max(0, min(patch_w - 1, cx + dx)), max(0, min(patch_h - 1, cy + dy))])
        pts = np.array(pts, dtype=np.int32)
        cv2.fillConvexPoly(shape, pts, 1)
    if shape.sum() == 0:
        cv2.circle(shape, (patch_w // 2, patch_h // 2), min(patch_h, patch_w) // 4, 1, -1)

    mask_bool = shape.astype(bool)
    patch_img = np.where(mask_bool[..., None], patch_img, 0)
    patch_mask = shape.astype(np.float32)

    angle = random.uniform(-30, 30)
    scale = random.uniform(0.8, 1.2)
    center = (patch_w / 2.0, patch_h / 2.0)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    patch_img = cv2.warpAffine(
        patch_img,
        rot_matrix,
        (patch_w, patch_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    patch_mask = cv2.warpAffine(
        patch_mask,
        rot_matrix,
        (patch_w, patch_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    ty1 = random.randint(0, h - patch_h)
    tx1 = random.randint(0, w - patch_w)
    ty2 = ty1 + patch_h
    tx2 = tx1 + patch_w

    target = image.copy()
    target[ty1:ty2, tx1:tx2] = patch_img

    if mask is None:
        mask = np.zeros((h, w), dtype=np.float32)
    elif mask.dtype != np.float32:
        mask = mask.astype(np.float32)

    mask[ty1:ty2, tx1:tx2] = np.clip(mask[ty1:ty2, tx1:tx2] + (patch_mask > 0).astype(np.float32), 0, 1)
    return target, mask


def multi_copy_move(image: np.ndarray, mask: Optional[np.ndarray], num_patches: int = 2, p: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """在一张图上重复做多次 copy-move，适用于弱增广场景。"""
    out_img, out_mask = image.copy(), mask.copy() if mask is not None else None
    for _ in range(num_patches):
        out_img, out_mask = synthetic_copy_move(out_img, out_mask, p=p)
    return out_img, out_mask
