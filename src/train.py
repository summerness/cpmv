import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from dataset import CopyMoveDataset
from augmentations import get_train_augmentations, get_valid_augmentations
from models.cmseg_lite import CMSegLite
from models.convnext_unetpp import ConvNeXt_UNetPP
from models.swin_deeplab import SwinDeepLabV3Plus
from utils.losses import multitask_loss
from utils.metrics import compute_f1

MODEL_FACTORY = {
    "convnext_unetpp": ConvNeXt_UNetPP,
    "swin_deeplab": SwinDeepLabV3Plus,
    "cmseg_lite": CMSegLite,
}


def seed_everything(seed: int = 42) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_paths(
    df: pd.DataFrame,
    img_col: str,
    mask_col: Optional[str],
    category_col: Optional[str] = None,
) -> Tuple[List[str], Optional[List[str]], Optional[List[Optional[str]]]]:
    imgs = df[img_col].tolist()
    masks = df[mask_col].tolist() if mask_col and mask_col in df.columns else None
    if masks is not None:
        processed_masks: List[str] = []
        for item in masks:
            if isinstance(item, str):
                processed_masks.append(item)
            elif pd.isna(item):
                processed_masks.append("")
            else:
                processed_masks.append(str(item))
        masks = processed_masks
    if category_col and category_col in df.columns:
        categories = df[category_col].tolist()
    else:
        categories = [None] * len(imgs)
    return imgs, masks, categories


def build_model(cfg: Dict) -> torch.nn.Module:
    name = cfg["model"]["name"].lower()
    params = cfg["model"].get("params", {})
    if name not in MODEL_FACTORY:
        raise ValueError(f"Unknown model: {name}")
    model = MODEL_FACTORY[name](**params)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(cfg["data"]["train_csv"])
    val_csv = cfg["data"].get("val_csv")
    if val_csv:
        val_df = pd.read_csv(val_csv)
    else:
        val_frac = cfg["data"].get("val_split", 0.1)
        val_df = train_df.sample(frac=val_frac, random_state=cfg.get("seed", 42))
        train_df = train_df.drop(val_df.index)

    image_size = tuple(cfg["data"].get("image_size", [512, 512]))
    train_aug = get_train_augmentations(
        image_size,
        use_elastic=cfg["data"].get("use_elastic", False),
        use_grid_distortion=cfg["data"].get("use_grid_distortion", False),
    )
    val_aug = get_valid_augmentations(image_size)

    category_col = cfg["data"].get("category_col", "category")
    train_imgs, train_masks, train_categories = load_paths(
        train_df,
        cfg["data"]["image_col"],
        cfg["data"].get("mask_col", "mask_path"),
        category_col=category_col,
    )
    val_imgs, val_masks, val_categories = load_paths(
        val_df,
        cfg["data"]["image_col"],
        cfg["data"].get("mask_col", "mask_path"),
        category_col=category_col,
    )

    train_dataset = CopyMoveDataset(
        train_imgs,
        train_masks,
        categories=train_categories,
        augment=train_aug,
        use_synthetic=cfg["data"].get("use_synthetic", False),
        synthetic_prob=cfg["data"].get("synthetic_prob", 0.25),
    )
    val_dataset = CopyMoveDataset(val_imgs, val_masks, categories=val_categories, augment=val_aug, use_synthetic=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"].get("batch_size", 4),
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"].get("val_batch_size", 2),
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
    )

    model = build_model(cfg)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"].get("lr", 1e-4), weight_decay=cfg["train"].get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["train"].get("epochs", 10),
        eta_min=cfg["train"].get("min_lr", 1e-6),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"].get("use_amp", True) and device.type == "cuda")

    save_dir = Path(cfg["train"].get("output_dir", "outputs"))
    save_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0

    for epoch in range(cfg["train"].get("epochs", 10)):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            cls_targets = (masks.view(masks.size(0), -1).max(dim=1)[0] > 0).float()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                mask_logits, cls_logits = model(images)
                total_loss, seg_loss, cls_loss = multitask_loss(
                    mask_logits,
                    masks,
                    cls_logits,
                    cls_targets,
                    cls_weight=cfg["train"].get("cls_weight", 0.3),
                )
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += total_loss.item()

        scheduler.step()

        val_loss = 0.0
        val_f1 = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                mask_logits, cls_logits = model(images)
                total_loss, _, _ = multitask_loss(
                    mask_logits,
                    masks,
                    cls_logits,
                    (masks.view(masks.size(0), -1).max(dim=1)[0] > 0).float(),
                    cls_weight=cfg["train"].get("cls_weight", 0.3),
                )
                val_loss += total_loss.item()
                val_f1 += compute_f1(mask_logits, masks)

        val_f1 /= max(len(val_loader), 1)
        avg_train_loss = running_loss / max(len(train_loader), 1)
        avg_val_loss = val_loss / max(len(val_loader), 1)

        print(
            f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt = {
                "model": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                "epoch": epoch + 1,
                "best_f1": best_f1,
                "config": cfg,
            }
            torch.save(ckpt, save_dir / "best.ckpt")
            print(f"Saved new best checkpoint with F1 {best_f1:.4f}")


if __name__ == "__main__":
    main()
