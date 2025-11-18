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
        A.VerticalFlip(p=0.25),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.2,
            rotate_limit=30,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.75,
        ),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
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


def synthetic_copy_move(image: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """随机制造一块伪造区域并更新 mask."""

    if image.ndim != 3:
        raise ValueError("Expect HWC RGB image for synthetic copy-move")

    h, w, _ = image.shape
    patch_h = random.randint(max(8, h // 12), max(12, h // 4))
    patch_w = random.randint(max(8, w // 12), max(12, w // 4))
    top = random.randint(0, max(0, h - patch_h))
    left = random.randint(0, max(0, w - patch_w))

    patch = image[top : top + patch_h, left : left + patch_w].copy()
    mask_patch = np.ones((patch_h, patch_w), dtype=np.uint8)

    angle = random.uniform(-60, 60)
    scale = random.uniform(0.8, 1.4)
    center = (patch_w / 2, patch_h / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    patch = cv2.warpAffine(
        patch,
        rot_matrix,
        (patch_w, patch_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    mask_patch = cv2.warpAffine(
        mask_patch,
        rot_matrix,
        (patch_w, patch_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    new_h, new_w = mask_patch.shape
    dest_top = random.randint(0, max(0, h - new_h))
    dest_left = random.randint(0, max(0, w - new_w))

    target = image.copy()
    target[dest_top : dest_top + new_h, dest_left : dest_left + new_w] = patch

    if mask is None:
        mask = np.zeros((h, w), dtype=np.uint8)
    else:
        mask = mask.copy()

    mask[dest_top : dest_top + new_h, dest_left : dest_left + new_w] = np.maximum(
        mask[dest_top : dest_top + new_h, dest_left : dest_left + new_w], mask_patch
    )

    return target, mask
