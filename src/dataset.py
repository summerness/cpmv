import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

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


def is_foreground(patch: np.ndarray, var_th: float = 8.0, lap_th: float = 6.0) -> bool:
    """判断 patch 是否有足够纹理/亮度方差，避免纯背景。"""
    if patch is None or patch.size == 0:
        return False
    gray = patch
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32)
    var = float(gray.var())
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_var = float(lap.var())
    return var > var_th and lap_var > lap_th


def generate_irregular_mask(h: int, w: int) -> np.ndarray:
    """生成不规则的源 mask，面积占比约 1%-10%。"""
    mask = np.zeros((h, w), dtype=np.uint8)
    area = h * w
    target_area = random.uniform(0.01, 0.1) * area
    max_radius = int(np.sqrt(target_area / np.pi))
    cx = random.randint(max_radius, max(w - max_radius, max_radius))
    cy = random.randint(max_radius, max(h - max_radius, max_radius))
    rx = random.randint(int(0.5 * max_radius), int(1.2 * max_radius))
    ry = random.randint(int(0.5 * max_radius), int(1.2 * max_radius))
    angle = random.randint(0, 179)
    cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 1, -1)
    # 形态扰动
    for _ in range(random.randint(1, 3)):
        k = random.randint(3, 7)
        if random.random() < 0.5:
            mask = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)
        else:
            mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)
    if random.random() < 0.3:
        mask = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0)
        mask = (mask > 0.5).astype(np.uint8)
    return mask


def _bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def sample_source_patch(image: np.ndarray, max_tries: int = 100):
    h, w = image.shape[:2]
    for _ in range(max_tries):
        mask_src = generate_irregular_mask(h, w)
        bbox = _bbox_from_mask(mask_src)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        src_patch = image[y0:y1, x0:x1]
        src_mask = mask_src[y0:y1, x0:x1]
        if is_foreground(src_patch):
            return src_patch, src_mask, (x0, y0, x1, y1)
    return None, None, None


def augment_patch_affine(patch: np.ndarray, mask: np.ndarray):
    h, w = mask.shape
    angle = random.uniform(-12, 12)
    scale = random.uniform(0.95, 1.1)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    patch_affine = cv2.warpAffine(patch, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask_affine = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if mask_affine.sum() < 0.3 * mask.sum():
        return patch, mask  # 变换过小则回退
    return patch_affine, mask_affine


def adjust_patch_to_background(patch: np.ndarray, bg: np.ndarray) -> np.ndarray:
    patch = patch.astype(np.float32)
    bg = bg.astype(np.float32)
    mean_p, std_p = patch.mean(axis=(0, 1)), patch.std(axis=(0, 1)) + 1e-6
    mean_b, std_b = bg.mean(axis=(0, 1)), bg.std(axis=(0, 1)) + 1e-6
    norm = (patch - mean_p) / std_p
    matched = norm * std_b + mean_b
    # 轻微亮度/噪声扰动
    matched = matched * (1 + random.uniform(-0.05, 0.05))
    gamma = random.uniform(0.9, 1.1)
    matched = np.clip(255.0 * ((matched / 255.0) ** gamma), 0, 255)
    if random.random() < 0.3:
        noise_std = random.uniform(2.0, 5.0)
        matched = matched + np.random.normal(0, noise_std, matched.shape)
    return np.clip(matched, 0, 255).astype(np.uint8)


def sample_target_location(image: np.ndarray, patch_mask: np.ndarray, src_bbox, max_tries: int = 20):
    H, W = image.shape[:2]
    ph, pw = patch_mask.shape
    sx0, sy0, sx1, sy1 = src_bbox
    src_cx, src_cy = (sx0 + sx1) / 2.0, (sy0 + sy1) / 2.0
    for _ in range(max_tries):
        tx = random.randint(0, max(0, W - pw))
        ty = random.randint(0, max(0, H - ph))
        cx, cy = tx + pw / 2.0, ty + ph / 2.0
        # 避免过近
        if abs(cx - src_cx) + abs(cy - src_cy) < 0.1 * (W + H):
            continue
        tgt_region = image[ty : ty + ph, tx : tx + pw]
        if is_foreground(tgt_region):
            return tx, ty
    return None, None


def blend_patch_to_image(image: np.ndarray, patch: np.ndarray, mask: np.ndarray, tx: int, ty: int):
    H, W = image.shape[:2]
    ph, pw = mask.shape
    forged = image.copy()
    forged_mask = np.zeros((H, W), dtype=np.uint8)
    # soft alpha
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (7, 7), 0)
    alpha = np.clip(alpha, 0, 1)[..., None]
    dst = forged[ty : ty + ph, tx : tx + pw]
    blended = patch.astype(np.float32) * alpha + dst.astype(np.float32) * (1 - alpha)
    forged[ty : ty + ph, tx : tx + pw] = blended.astype(np.uint8)
    forged_mask[ty : ty + ph, tx : tx + pw] = mask
    return forged, forged_mask


def synthetic_forgery(image: np.ndarray, base_mask: Optional[np.ndarray], p: float) -> tuple[np.ndarray, np.ndarray]:
    """生成单次 copy-move 伪造。"""
    if random.random() > p:
        return image, base_mask
    h, w = image.shape[:2]
    src_patch, src_mask, bbox = sample_source_patch(image)
    if src_patch is None:
        return image, base_mask
    src_patch, src_mask = augment_patch_affine(src_patch, src_mask)
    sx0, sy0, sx1, sy1 = bbox
    ph, pw = src_mask.shape
    # 选择目标位置
    tx, ty = sample_target_location(image, src_mask, bbox)
    if tx is None:
        return image, base_mask
    # 颜色匹配
    tgt_region = image[ty : ty + ph, tx : tx + pw]
    src_patch = adjust_patch_to_background(src_patch, tgt_region)
    forged, forged_mask = blend_patch_to_image(image, src_patch, src_mask, tx, ty)
    # 合并 mask
    if base_mask is None:
        base_mask = np.zeros((h, w), dtype=np.uint8)
    if base_mask.shape != forged_mask.shape:
        base_mask = cv2.resize(base_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    final_mask = np.clip(base_mask + forged_mask, 0, 1).astype(np.uint8)
    return forged, final_mask


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
        synthetic_on_base: bool = False,
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
        self.synthetic_times = max(1, synthetic_times)  # 最大次数
        self.synthetic_copies = max(0, synthetic_copies)
        self.synthetic_on_base = synthetic_on_base
        self.base_len = len(self.image_paths)
        # 记录 authentic 样本索引用于生成 copy-move 样本
        self.authentic_indices = []
        if self.categories is not None:
            self.authentic_indices = [i for i, cat in enumerate(self.categories) if cat == "authentic"]
        elif self.mask_paths is not None:
            for i, m in enumerate(self.mask_paths):
                if m is None or str(m).strip() == "":
                    self.authentic_indices.append(i)
        # 为合成样本预构建池：只复制 authentic 索引
        self.synthetic_pool: List[int] = []
        if self.synthetic_copies > 0 and self.authentic_indices:
            for _ in range(self.synthetic_copies):
                self.synthetic_pool.extend(self.authentic_indices)
        self._warn_missing = True

    def __len__(self) -> int:
        return self.base_len + len(self.synthetic_pool)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        is_syn_dup = idx >= self.base_len
        if is_syn_dup:
            syn_idx = idx - self.base_len
            if not self.synthetic_pool:
                raise IndexError("Synthetic pool is empty but synthetic index requested.")
            base_idx = self.synthetic_pool[syn_idx % len(self.synthetic_pool)]
        else:
            base_idx = idx

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

        # 合成伪造：可以作用于合成副本或原始样本（取决于配置）
        def apply_synthetic(img, msk) -> tuple[np.ndarray, np.ndarray]:
            forged_img, forged_m = img, msk
            max_times = self.synthetic_times
            times = random.randint(1, max_times) if max_times > 1 else 1
            for _ in range(times):
                forged_img, forged_m = synthetic_forgery(
                    forged_img,
                    forged_m if forged_m is not None else np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8),
                    p=self.synthetic_prob if self.use_synthetic else 0.0,
                )
            return forged_img, forged_m

        if is_syn_dup:
            # 仅使用 authentic 池复制出的样本：无论是否已有 mask，都按概率伪造
            image, mask = apply_synthetic(image, mask)
        elif self.use_synthetic and self.synthetic_on_base:
            # 原始样本也可以按概率注入伪造（仅在 forged_ratio 很低时）
            if forged_ratio < 0.01:
                image, mask = apply_synthetic(image, mask)

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
            cat = self.categories[base_idx]
            sample["category"] = "forged" if is_syn_dup else cat
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
