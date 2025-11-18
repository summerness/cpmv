import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentations import synthetic_copy_move


def _read_mask(mask_path: Path) -> np.ndarray:
    ext = mask_path.suffix.lower()
    if ext == ".npy":
        mask = np.load(mask_path)
    elif ext == ".npz":
        with np.load(mask_path) as data:
            key = list(data.files)[0]
            mask = data[key]
    else:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Unable to read mask: {mask_path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0.5).astype(np.uint8)
    return mask


class CopyMoveDataset(Dataset):
    """Dataset that optionally applies copy-move synthesis and augmentations."""

    def __init__(
        self,
        image_paths: Sequence[Path],
        mask_paths: Optional[Sequence[Path]] = None,
        augment: Optional[Callable] = None,
        use_synthetic: bool = False,
        synthetic_prob: float = 0.25,
    ) -> None:
        if mask_paths is not None and len(image_paths) != len(mask_paths):
            raise ValueError("image_paths and mask_paths must have identical length")

        self.image_paths = [Path(p) for p in image_paths]
        self.mask_paths = [Path(p) for p in mask_paths] if mask_paths is not None else None
        self.augment = augment
        self.use_synthetic = use_synthetic
        self.synthetic_prob = synthetic_prob

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = None
        if self.mask_paths is not None:
            mask_path = self.mask_paths[idx]
            if mask_path.suffix.lower() in {".npy", ".npz"}:
                mask = _read_mask(mask_path)
            else:
                mask = _read_mask(mask_path)

        if self.use_synthetic and mask is not None:
            forged_ratio = mask.mean() if mask.size else 0.0
            if forged_ratio < 0.01 and random.random() < self.synthetic_prob:
                image, mask = synthetic_copy_move(image, mask)

        data = {"image": image}
        if mask is not None:
            data["mask"] = mask

        if self.augment is not None:
            augmented = self.augment(**data)
        else:
            augmented = data

        image_tensor = self._to_tensor(augmented["image"])
        mask_tensor = None
        if "mask" in augmented:
            mask_tensor = self._to_tensor(augmented["mask"], is_mask=True)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "id": image_path.stem,
        }

    @staticmethod
    def _to_tensor(arr, is_mask: bool = False) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            tensor = arr.float()
            return tensor if not is_mask else tensor

        if arr.ndim == 2 and not is_mask:
            arr = np.expand_dims(arr, axis=-1)

        if arr.ndim == 3 and arr.shape[0] != 3 and arr.shape[-1] == 3 and not is_mask:
            arr = np.transpose(arr, (2, 0, 1))

        if is_mask and arr.ndim == 2:
            tensor = torch.from_numpy(arr).unsqueeze(0).float()
        else:
            tensor = torch.from_numpy(arr).float()

        if not is_mask:
            tensor = tensor / 255.0 if tensor.max() > 1.0 else tensor

        return tensor
