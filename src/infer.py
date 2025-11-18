import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from augmentations import get_valid_augmentations
from models.cmseg_lite import CMSegLite
from models.convnext_unetpp import ConvNeXt_UNetPP
from models.swin_deeplab import SwinDeepLabV3Plus
from utils.postprocess import postprocess_mask
from utils.rle import rle_encode

MODEL_FACTORY = {
    "convnext_unetpp": ConvNeXt_UNetPP,
    "swin_deeplab": SwinDeepLabV3Plus,
    "cmseg_lite": CMSegLite,
}


class InferenceDataset(Dataset):
    def __init__(self, image_paths: List[Path], augment) -> None:
        self.image_paths = [Path(p) for p in image_paths]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        transformed = self.augment(image=image)
        tensor = transformed["image"]
        return tensor, path.stem, (h, w)


def build_model(name: str, params: dict) -> torch.nn.Module:
    if name not in MODEL_FACTORY:
        raise ValueError(f"Unknown model {name}")
    return MODEL_FACTORY[name](**params)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--model-name", type=str, default=None, help="Override checkpoint model name if needed")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, help="Override checkpoint resize if needed")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--area-threshold", type=int, default=256)
    parser.add_argument("--cls-threshold", type=float, default=0.5)
    parser.add_argument("--save-prob", action="store_true")
    parser.add_argument("--prob-dir", type=str, default="probabilities")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = checkpoint.get("config", {})
    model_name = args.model_name or cfg.get("model", {}).get("name", "convnext_unetpp")
    image_size = tuple(args.image_size or cfg.get("data", {}).get("image_size", [512, 512]))
    model_params = cfg.get("model", {}).get("params", {})
    model_params.setdefault("num_classes", 1)
    model = build_model(model_name, model_params)
    model.load_state_dict(checkpoint["model"], strict=False)
    model = model.to(device)
    model.eval()

    image_dir = Path(args.image_dir)
    image_paths = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    augment = get_valid_augmentations(image_size)
    dataset = InferenceDataset(image_paths, augment)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    rows = []
    cls_records = []
    prob_dir = Path(args.prob_dir)
    if args.save_prob:
        prob_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for images, image_ids, shapes in loader:
            images = images.to(device)
            mask_logits, cls_logits = model(images)
            prob = torch.sigmoid(mask_logits).cpu().numpy()[0, 0]
            cls_prob = torch.sigmoid(cls_logits).cpu().numpy().reshape(-1)[0]
            h, w = shapes[0]
            resized_prob = cv2.resize(prob, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
            mask = postprocess_mask(resized_prob, args.threshold, args.area_threshold)
            if mask.sum() < args.area_threshold or cls_prob < args.cls_threshold:
                annotation = "authentic"
            else:
                annotation = rle_encode(mask)
            rows.append({"case_id": image_ids[0], "annotation": annotation})
            cls_records.append({"case_id": image_ids[0], "cls_prob": cls_prob})
            if args.save_prob:
                np.save(prob_dir / f"{image_ids[0]}.npy", resized_prob)

    import pandas as pd

    pd.DataFrame(rows).to_csv(args.output, index=False)
    if args.save_prob:
        pd.DataFrame(cls_records).to_csv(prob_dir / "cls_probs.csv", index=False)


if __name__ == "__main__":
    main()
