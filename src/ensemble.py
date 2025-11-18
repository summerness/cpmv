import argparse
import importlib
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch

from augmentations import get_valid_augmentations
from utils.postprocess import postprocess_mask
from utils.rle import rle_encode


def parse_args():
    parser = argparse.ArgumentParser(description="Run 5-model ensemble inference.")
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="List of checkpoint paths (expected 5).")
    parser.add_argument("--model-names", nargs="*", default=None, help="Optional override fully-qualified model targets matching checkpoints.")
    parser.add_argument("--weights", nargs="*", type=float, default=None, help="Optional weights for checkpoints.")
    parser.add_argument("--output", type=str, default="ensemble.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--area-threshold", type=int, default=256)
    parser.add_argument("--cls-threshold", type=float, default=0.5)
    parser.add_argument("--save-prob-dir", type=str, default=None, help="Optional directory to save ensembled probability maps.")
    return parser.parse_args()


def build_model(target: str, params: dict) -> torch.nn.Module:
    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**params)


def load_checkpoint(path: Path, device: torch.device, override_target: str = None):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config", {})
    model_target = override_target or cfg.get("model", {}).get("target")
    if model_target is None:
        raise ValueError(f"Checkpoint {path} missing model target in config; please provide --model-names override.")
    params = cfg.get("model", {}).get("params", {})
    params.setdefault("num_classes", 1)
    model = build_model(model_target, params)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device)
    model.eval()
    image_size = tuple(cfg.get("data", {}).get("image_size", [512, 512]))
    val_aug = cfg.get("augmentations", {}).get("val")
    return model, image_size, val_aug


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_names and len(args.model_names) != len(args.checkpoints):
        raise ValueError("--model-names must match number of checkpoints")

    if len(args.checkpoints) != 5:
        print(f"Warning: expected 5 checkpoints, received {len(args.checkpoints)}")

    weights = args.weights or [1.0] * len(args.checkpoints)
    if len(weights) != len(args.checkpoints):
        raise ValueError("--weights length must match checkpoints")
    weight_sum = sum(weights)

    models: List[torch.nn.Module] = []
    transforms = []
    for idx, ckpt_path in enumerate(args.checkpoints):
        target_override = args.model_names[idx] if args.model_names else None
        model, image_size, val_aug = load_checkpoint(Path(ckpt_path), device, override_target=target_override)
        models.append(model)
        transforms.append(get_valid_augmentations(image_size, val_aug))

    image_dir = Path(args.image_dir)
    image_paths = sorted([p for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff") for p in image_dir.glob(ext)])
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    prob_dir = Path(args.save_prob_dir) if args.save_prob_dir else None
    if prob_dir:
        prob_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for image_path in image_paths:
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                raise FileNotFoundError(image_path)
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]

            prob_sum = np.zeros((h, w), dtype=np.float32)
            cls_sum = 0.0

            for model, transform, weight in zip(models, transforms, weights):
                tensor = transform(image=image)["image"].unsqueeze(0).to(device)
                mask_logits, cls_logits = model(tensor)
                prob = torch.sigmoid(mask_logits).cpu().numpy()[0, 0]
                prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
                prob_sum += weight * prob
                cls_prob = torch.sigmoid(cls_logits).item()
                cls_sum += weight * cls_prob

            mean_prob = prob_sum / weight_sum
            mean_cls = cls_sum / weight_sum
            mask = postprocess_mask(mean_prob, args.threshold, args.area_threshold)

            if mask.sum() < args.area_threshold or mean_cls < args.cls_threshold:
                annotation = "authentic"
            else:
                annotation = rle_encode(mask)

            rows.append({"case_id": image_path.stem, "annotation": annotation})
            if prob_dir:
                np.save(prob_dir / f"{image_path.stem}.npy", mean_prob)

    pd.DataFrame(rows).sort_values("case_id").to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
