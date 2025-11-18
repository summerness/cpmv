import random
from typing import Optional, Tuple

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


def get_train_augmentations(
    image_size: Tuple[int, int],
    use_elastic: bool = False,
    use_grid_distortion: bool = False,
    use_synthetic: bool = False,
) -> A.Compose:
    h, w = image_size
    transforms = [
        A.Resize(height=h, width=w),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=30,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
    ]

    if use_elastic:
        transforms.append(A.ElasticTransform(p=0.2, alpha=80, sigma=12, alpha_affine=10))
    if use_grid_distortion:
        transforms.append(A.GridDistortion(p=0.2))

    transforms.extend(
        [
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms, additional_targets={"mask": "mask"})


def get_valid_augmentations(image_size: Tuple[int, int]) -> A.Compose:
    h, w = image_size
    return A.Compose(
        [
            A.Resize(height=h, width=w),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
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
    patch_h = random.randint(int(0.1 * h), max(int(0.3 * h), 1))
    patch_w = random.randint(int(0.1 * w), max(int(0.3 * w), 1))
    if patch_h <= 0 or patch_w <= 0:
        return image, mask

    y1 = random.randint(0, h - patch_h)
    x1 = random.randint(0, w - patch_w)
    y2 = y1 + patch_h
    x2 = x1 + patch_w

    patch_img = image[y1:y2, x1:x2].copy()
    patch_mask = mask[y1:y2, x1:x2].copy() if mask is not None else np.zeros((patch_h, patch_w), dtype=np.float32)

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
        mask = np.zeros((h, w), dtype=np.uint8)
    elif mask.dtype != np.float32:
        mask = mask.astype(np.float32)

    mask[ty1:ty2, tx1:tx2] = np.clip(mask[ty1:ty2, tx1:tx2] + (patch_mask > 0).astype(np.float32), 0, 1)
    return target, mask
