import argparse
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from augmentations import get_train_augmentations, get_valid_augmentations
from dataset import CopyMoveDataset
from utils.losses import multitask_loss
from utils.metrics import compute_f1


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
    model_cfg = cfg.get("model", {})
    target = model_cfg.get("target")
    if not target:
        raise ValueError("model.target must be provided in config")
    params = model_cfg.get("params", {})
    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    model = cls(**params)
    return model


def split_train_val(df: pd.DataFrame, data_cfg: Dict, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fold_col = data_cfg.get("fold_col")
    fold_id = data_cfg.get("fold_id")
    if fold_col and fold_col in df.columns and fold_id is not None:
        val_df = df[df[fold_col] == fold_id]
        train_df = df[df[fold_col] != fold_id]
        if val_df.empty:
            raise ValueError(f"No validation samples found for fold {fold_id}")
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    val_csv = data_cfg.get("val_csv")
    if val_csv:
        val_df = pd.read_csv(val_csv)
        train_df = df
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    val_frac = data_cfg.get("val_split", 0.1)
    if val_frac <= 0 or val_frac >= 1:
        raise ValueError("val_split must be between 0 and 1 when no fold/val_csv is provided")
    val_df = df.sample(frac=val_frac, random_state=seed)
    train_df = df.drop(val_df.index)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_df = pd.read_csv(cfg["data"]["train_csv"])
    train_df, val_df = split_train_val(full_df, cfg["data"], cfg.get("seed", 42))

    image_size = tuple(cfg["data"].get("image_size", [512, 512]))
    aug_cfg = cfg.get("augmentations", {})
    train_aug = get_train_augmentations(image_size, aug_cfg.get("train"))
    val_aug = get_valid_augmentations(image_size, aug_cfg.get("val"))

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
    val_dataset = CopyMoveDataset(
        val_imgs,
        val_masks,
        categories=val_categories,
        augment=val_aug,
        use_synthetic=False,
    )

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
    scheduler_cfg = cfg["train"].get("scheduler", {"name": "cosine"})
    scheduler_name = scheduler_cfg.get("name", "cosine").lower()
    epochs = cfg["train"].get("epochs", 10)
    if scheduler_name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg["train"].get("lr", 1e-4),
            epochs=epochs,
            steps_per_epoch=max(len(train_loader), 1),
            pct_start=scheduler_cfg.get("pct_start", 0.3),
            div_factor=scheduler_cfg.get("div_factor", 25.0),
            final_div_factor=scheduler_cfg.get("final_div_factor", 1e4),
        )
        step_per_batch = True
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=cfg["train"].get("min_lr", 1e-6),
        )
        step_per_batch = False

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"].get("use_amp", True) and device.type == "cuda")

    save_dir = Path(cfg["train"].get("output_dir", "outputs"))
    save_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
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
            if step_per_batch:
                scheduler.step()

            running_loss += total_loss.item()

        if not step_per_batch:
            scheduler.step()

        val_loss = 0.0
        val_f1 = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)
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
