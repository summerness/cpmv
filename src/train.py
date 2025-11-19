import argparse
import importlib
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
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


def setup_logger(output_dir: Path) -> logging.Logger:
    logger_name = f"trainer_{output_dir}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "train.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


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
    parser.add_argument("--fold-id", type=int, default=None, help="Override data.fold_id for cross-validation runs.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.fold_id is not None:
        cfg.setdefault("data", {})["fold_id"] = args.fold_id

    data_cfg = cfg.get("data", {})
    fold_col = data_cfg.get("fold_col")
    specified_fold = data_cfg.get("fold_id")
    if specified_fold is not None:
        run_training(cfg)
        return

    if fold_col and Path(data_cfg["train_csv"]).exists():
        df = pd.read_csv(data_cfg["train_csv"], usecols=[fold_col])
        folds = sorted(df[fold_col].dropna().unique().tolist())
        if folds:
            for fid in folds:
                cfg_fold = deepcopy(cfg)
                cfg_fold.setdefault("data", {})["fold_id"] = int(fid)
                print(f"=== Training fold {fid} ===")
                run_training(cfg_fold)
            return

    run_training(cfg)


def run_training(cfg: Dict) -> None:
    seed_everything(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_df = pd.read_csv(cfg["data"]["train_csv"])
    train_df, val_df = split_train_val(full_df, cfg["data"], cfg.get("seed", 42))

    save_dir = Path(cfg["train"].get("output_dir", "outputs"))
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(save_dir)
    logger.info("Using device: %s", device)
    logger.info("Active fold: %s", cfg["data"].get("fold_id", "N/A"))
    logger.info("Training samples: %d | Validation samples: %d", len(train_df), len(val_df))
    mask_col = cfg["data"].get("mask_col", "mask_path")
    if mask_col in train_df.columns:
        train_ratio = (train_df[mask_col].fillna("").astype(str).str.len() > 0).mean()
        val_ratio = (val_df[mask_col].fillna("").astype(str).str.len() > 0).mean()
    else:
        train_ratio = val_ratio = 0.0
    logger.info("Forged sample ratio -> train: %.4f val: %.4f", train_ratio, val_ratio)
    if val_ratio == 0:
        logger.warning("Validation set contains no forged masks; F1 will stay at 0 until positives appear.")

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
    logger.info("Train augmentation config: %s", aug_cfg.get("train", {}))

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

    best_f1 = 0.0
    train_cfg = cfg["train"]
    log_every = train_cfg.get("log_every", 50)
    debug_cfg = train_cfg.get("debug", {})
    debug_val_batches = debug_cfg.get("val_batches", train_cfg.get("debug_val_batches", 0))
    save_debug_samples = debug_cfg.get("save_samples", False)
    debug_sample_limit = debug_cfg.get("max_samples", 0)
    debug_dir = Path(debug_cfg.get("output_dir", save_dir / "debug"))
    debug_saved = 0
    logger.info(
        "Start training for %d epochs | batch_size=%d | steps_per_epoch=%d",
        epochs,
        cfg["train"].get("batch_size", 4),
        len(train_loader),
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
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
            if log_every and (step + 1) % log_every == 0:
                logger.info(
                    "Epoch %d | Step %d/%d | total=%.4f seg=%.4f cls=%.4f",
                    epoch + 1,
                    step + 1,
                    len(train_loader),
                    total_loss.item(),
                    seg_loss.item(),
                    cls_loss.item(),
                )

        if not step_per_batch:
            scheduler.step()

        val_loss = 0.0
        val_f1 = 0.0
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)
                mask_logits, cls_logits = model(images)
                if step < debug_val_batches:
                    probs = torch.sigmoid(mask_logits)
                    logger.info(
                        "Val debug e%02d b%03d -> probs mean %.4f max %.4f | mask mean %.4f",
                        epoch + 1,
                        step + 1,
                        probs.mean().item(),
                        probs.max().item(),
                        masks.float().mean().item(),
                    )
                    if save_debug_samples and debug_saved < debug_sample_limit:
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        ids = batch.get("id", [f"{i}" for i in range(len(images))])
                        probs_np = probs.detach().cpu().numpy()
                        masks_np = masks.detach().cpu().numpy()
                        for idx in range(probs_np.shape[0]):
                            if debug_saved >= debug_sample_limit:
                                break
                            sample_id = ids[idx] if idx < len(ids) else f"{idx}"
                            prefix = f"e{epoch+1:02d}_b{step+1:03d}_{sample_id}"
                            np.save(debug_dir / f"{prefix}_prob.npy", probs_np[idx, 0])
                            np.save(debug_dir / f"{prefix}_mask.npy", masks_np[idx, 0])
                            debug_saved += 1
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

        logger.info(
            "Epoch %d complete | train_loss=%.4f val_loss=%.4f val_f1=%.4f",
            epoch + 1,
            avg_train_loss,
            avg_val_loss,
            val_f1,
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
            logger.info("Saved new best checkpoint with F1 %.4f", best_f1)


if __name__ == "__main__":
    main()
