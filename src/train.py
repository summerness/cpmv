import argparse
import importlib
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from augmentations import get_train_augmentations, get_valid_augmentations
from dataset import CopyMoveDataset
from utils.losses import MultiTaskLoss, SegmentationLoss
from utils.metrics import compute_f1
from utils.postprocess import postprocess_mask


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

    data_cfg = cfg["data"]
    full_df = pd.read_csv(data_cfg["train_csv"])
    overfit_n = data_cfg.get("overfit_n", 0)
    if overfit_n and overfit_n > 0:
        train_df = full_df.sample(
            n=overfit_n,
            random_state=cfg.get("seed", 42),
            replace=overfit_n > len(full_df),
        ).reset_index(drop=True)
        val_df = train_df.copy()
    else:
        train_df, val_df = split_train_val(full_df, data_cfg, cfg.get("seed", 42))

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
        synthetic_times=cfg["data"].get("synthetic_times", 1),
        synthetic_copies=cfg["data"].get("synthetic_copies", 0),
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
    multitask_wrapper = MultiTaskLoss()

    best_f1 = 0.0
    train_cfg = cfg["train"]
    log_every = train_cfg.get("log_every", 50)
    debug_cfg = train_cfg.get("debug", {})
    log_train_f1 = train_cfg.get("log_train_f1", False)
    max_train_f1_batches = train_cfg.get("train_f1_batches")
    sweep_cfg = train_cfg.get("eval_sweep", {})
    def _expand_range(val):
        if isinstance(val, (list, tuple)) and len(val) == 3:
            start, stop, step = val
            return list(np.arange(start, stop + 1e-8, step))
        return val if isinstance(val, (list, tuple)) else []

    sweep_thresholds = _expand_range(sweep_cfg.get("thresholds", []))
    sweep_areas = _expand_range(sweep_cfg.get("areas", []))
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
                # 统一 cls logits 形状为 [B, 2] 用于 CE
                if cls_logits.dim() == 1:
                    cls_logits = cls_logits.unsqueeze(1)
                if cls_logits.shape[1] == 1:
                    cls_logits_ce = torch.cat([-cls_logits, cls_logits], dim=1)
                else:
                    cls_logits_ce = cls_logits
                seg_loss = SegmentationLoss()(mask_logits, masks)
                cls_loss = F.cross_entropy(cls_logits_ce, cls_targets.long().view(-1))
                total_loss = multitask_wrapper(seg_loss, cls_loss)
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
        sweep_probs = [] if sweep_thresholds and sweep_areas else None
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)
                mask_logits, cls_logits = model(images)
                val_seg_loss = SegmentationLoss()(mask_logits, masks)
                if cls_logits.dim() == 1:
                    cls_logits = cls_logits.unsqueeze(1)
                val_cls_targets = (masks.view(masks.size(0), -1).max(dim=1)[0] > 0).float()
                if cls_logits.shape[1] == 1:
                    cls_logits_ce = torch.cat([-cls_logits, cls_logits], dim=1)
                else:
                    cls_logits_ce = cls_logits
                val_cls_loss = F.cross_entropy(cls_logits_ce, val_cls_targets.long().view(-1))
                total_loss = multitask_wrapper(val_seg_loss, val_cls_loss)
                val_loss += total_loss.item()
                val_f1 += compute_f1(mask_logits, masks)
                if sweep_probs is not None:
                    sweep_probs.append((torch.sigmoid(mask_logits).detach().cpu(), masks.detach().cpu()))

        train_f1 = None
        if log_train_f1:
            f1_accum, count = 0.0, 0
            model.eval()
            with torch.no_grad():
                for b_idx, batch in enumerate(train_loader):
                    if max_train_f1_batches is not None and b_idx >= max_train_f1_batches:
                        break
                    imgs = batch["image"].to(device)
                    msks = batch["mask"].to(device)
                    if msks.ndim == 3:
                        msks = msks.unsqueeze(1)
                    logits, _ = model(imgs)
                    f1_accum += compute_f1(logits, msks)
                    count += 1
            train_f1 = f1_accum / max(count, 1)

        val_f1 /= max(len(val_loader), 1)
        avg_train_loss = running_loss / max(len(train_loader), 1)
        avg_val_loss = val_loss / max(len(val_loader), 1)

        if sweep_probs is not None:
            def f1_np(pred, gt):
                pred = pred.astype(np.uint8)
                gt = gt.astype(np.uint8)
                tp = (pred & gt).sum()
                fp = (pred & (1 - gt)).sum()
                fn = ((1 - pred) & gt).sum()
                precision = tp / (tp + fp + 1e-7)
                recall = tp / (tp + fn + 1e-7)
                return 2 * precision * recall / (precision + recall + 1e-7)

            best_combo = None
            best_score = -1
            for thr in sweep_thresholds:
                for area in sweep_areas:
                    scores = []
                    for probs, msks in sweep_probs:
                        prob_np = probs.numpy()
                        msk_np = msks.numpy()
                        for b in range(prob_np.shape[0]):
                            mask_pp = postprocess_mask(prob_np[b, 0], thr, area)
                            scores.append(f1_np(mask_pp, (msk_np[b, 0] > 0.5).astype(np.uint8)))
                    score = float(np.mean(scores)) if scores else 0.0
                    if score > best_score:
                        best_score = score
                        best_combo = (thr, area)
            if best_combo is not None:
                logger.info("Sweep best -> thr=%.3f area=%d f1=%.4f", best_combo[0], best_combo[1], best_score)

        logger.info(
            "Epoch %d complete | train_loss=%.4f val_loss=%.4f val_f1=%.4f%s",
            epoch + 1,
            avg_train_loss,
            avg_val_loss,
            val_f1,
            "" if train_f1 is None else f" train_f1={train_f1:.4f}",
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
