import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from augmentations import synthetic_copy_move


def _read_mask(mask_path) -> np.ndarray:
    """支持单个路径或 'path1|path2' 形式的多路径，返回合并后的二值 mask."""

    def _load_single(p: Path) -> np.ndarray:
        ext = p.suffix.lower()
        if ext == ".npy":
            m = np.load(p)
        elif ext == ".npz":
            with np.load(p) as data:
                key = list(data.files)[0]
                m = data[key]
        else:
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise FileNotFoundError(f"Unable to read mask: {p}")
        if m.ndim == 3:
            # 如果 npy/npz 存成 [N,H,W] 或多通道，取最大值聚合
            if m.shape[0] not in (m.shape[1], m.shape[2]):
                m = m.max(axis=0)
            else:
                m = m[..., 0]
        return (m != 0).astype(np.uint8)

    masks = []
    if isinstance(mask_path, (str, Path)):
        parts = str(mask_path).split("|") if isinstance(mask_path, str) else [mask_path]
    else:
        parts = list(mask_path)

    for part in parts:
        part = Path(part)
        if not str(part).strip():
            continue
        if not part.exists():
            continue
        masks.append(_load_single(Path(part)))

    if not masks:
        raise FileNotFoundError(f"No valid mask files provided: {mask_path}")
    mask = masks[0]
    for m in masks[1:]:
        if m.shape != mask.shape:
            m = cv2.resize(m, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = np.maximum(mask, m)
    return mask


class CopyMoveDataset(Dataset):
    """Dataset that optionally applies copy-move synthesis and augmentations."""

    def __init__(
        self,
        image_paths: Sequence[Path],
        mask_paths: Optional[Sequence[Path]] = None,
        categories: Optional[Sequence[Optional[str]]] = None,
        augment: Optional[Callable] = None,
        use_synthetic: bool = False,
        synthetic_prob: float = 0.25,
        synthetic_times: int = 1,
        synthetic_copies: int = 0,
    ) -> None:
        if mask_paths is not None and len(image_paths) != len(mask_paths):
            raise ValueError("image_paths and mask_paths must have identical length")
        if categories is not None and len(categories) != len(image_paths):
            raise ValueError("categories must have identical length as image_paths")

        self.image_paths = [Path(p) for p in image_paths]
        self.mask_paths = list(mask_paths) if mask_paths is not None else None
        self.categories = list(categories) if categories is not None else None
        self.augment = augment
        self.use_synthetic = use_synthetic
        self.synthetic_prob = synthetic_prob
        self.synthetic_times = max(1, synthetic_times)
        self.synthetic_copies = max(0, synthetic_copies)
        self.base_len = len(self.image_paths)
        # 记录 authentic 样本索引用于生成 copy-move 样本
        self.authentic_indices = []
        if self.mask_paths is not None:
            for i, m in enumerate(self.mask_paths):
                if m is None or str(m).strip() == "":
                    self.authentic_indices.append(i)

    def __len__(self) -> int:
        return self.base_len * (1 + self.synthetic_copies)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base_idx = idx % self.base_len
        is_syn_dup = idx >= self.base_len
        if is_syn_dup and self.authentic_indices:
            base_idx = random.choice(self.authentic_indices)

        image_path = self.image_paths[base_idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = None
        if self.mask_paths is not None:
            mask_entry = self.mask_paths[base_idx]
            if mask_entry is None or str(mask_entry).strip() == "":
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
            else:
                try:
                    mask = _read_mask(mask_entry)
                except FileNotFoundError:
                    if getattr(self, "_warn_missing", True):
                        print(f"[CopyMoveDataset] Missing mask files for {mask_entry}; using zeros.")
                        self._warn_missing = False
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        forged_ratio = float(mask.mean()) if mask is not None and mask.size else 0.0

        if is_syn_dup:
            if forged_ratio < 0.01:
                if mask is None:
                    mask = np.zeros(image.shape[:2], dtype=np.float32)
                for _ in range(self.synthetic_times):
                    image, mask = synthetic_copy_move(image, mask, p=1.0)

        elif self.use_synthetic:
            if forged_ratio < 0.01:
                for _ in range(self.synthetic_times):
                    if random.random() < self.synthetic_prob:
                        image, mask = synthetic_copy_move(
                            image,
                            mask if mask is not None else np.zeros((image.shape[0], image.shape[1]), dtype=np.float32),
                            p=1.0,
                        )

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

        sample = {
            "image": image_tensor,
            "mask": mask_tensor,
            "id": image_path.stem,
        }
        if self.categories is not None:
            sample["category"] = self.categories[base_idx]
        return sample

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
