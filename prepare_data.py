import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
MASK_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy", ".npz")


def list_images(directory: Path) -> List[Path]:
    files = []
    for ext in IMAGE_EXTS:
        files.extend(directory.glob(f"*{ext}"))
    return sorted(files)


def find_mask(mask_dir: Optional[Path], stem: str) -> Optional[Path]:
    if mask_dir is None:
        return None
    for ext in MASK_EXTS:
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def build_train_df(
    image_dir: Path,
    mask_dir: Path,
    source_label: str,
    allow_missing_mask: bool = False,
) -> pd.DataFrame:
    rows = []
    for image_path in list_images(image_dir):
        stem = image_path.stem
        mask_path = find_mask(mask_dir, stem)
        if mask_path is None:
            if allow_missing_mask:
                continue
            raise FileNotFoundError(f"Mask for {image_path} not found in {mask_dir}")
        rows.append({
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "source": source_label,
        })
    return pd.DataFrame(rows)


def build_test_csv(image_dir: Path, output: Path, sample_submission: Optional[Path] = None) -> None:
    if sample_submission and sample_submission.exists():
        sub_df = pd.read_csv(sample_submission)
        rows = []
        for case_id in sub_df["case_id"].astype(str):
            for ext in IMAGE_EXTS:
                candidate = image_dir / f"{case_id}{ext}"
                if candidate.exists():
                    rows.append({"image_path": str(candidate), "case_id": case_id})
                    break
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame({
            "image_path": [str(p) for p in list_images(image_dir)],
        })
        df["case_id"] = df["image_path"].apply(lambda p: Path(p).stem)
    df.to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CSV metadata from directory structure.")
    parser.add_argument("--train-images", type=str, required=True)
    parser.add_argument("--train-masks", type=str, required=True)
    parser.add_argument("--train-output", type=str, default="data/train.csv")
    parser.add_argument("--supp-images", type=str, default=None)
    parser.add_argument("--supp-masks", type=str, default=None)
    parser.add_argument("--include-supp", action="store_true")
    parser.add_argument("--test-images", type=str, default=None)
    parser.add_argument("--test-output", type=str, default="data/test.csv")
    parser.add_argument("--sample-submission", type=str, default=None)
    parser.add_argument("--allow-missing-mask", action="store_true")
    args = parser.parse_args()

    train_dir = Path(args.train_images)
    train_mask_dir = Path(args.train_masks)
    train_output = Path(args.train_output)
    train_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building train CSV from {train_dir} and {train_mask_dir}")
    train_df = build_train_df(train_dir, train_mask_dir, source_label="primary", allow_missing_mask=args.allow_missing_mask)

    if args.include_supp and args.supp_images and args.supp_masks:
        supp_dir = Path(args.supp_images)
        supp_mask_dir = Path(args.supp_masks)
        supp_df = build_train_df(
            supp_dir,
            supp_mask_dir,
            source_label="supplemental",
            allow_missing_mask=args.allow_missing_mask,
        )
        combined = pd.concat([train_df, supp_df], ignore_index=True)
    else:
        combined = train_df

    combined.to_csv(train_output, index=False)
    print(f"Saved {len(combined)} rows to {train_output}")

    if args.test_images:
        test_dir = Path(args.test_images)
        test_output = Path(args.test_output)
        test_output.parent.mkdir(parents=True, exist_ok=True)
        sample_sub = Path(args.sample_submission) if args.sample_submission else None
        build_test_csv(test_dir, test_output, sample_sub)
        print(f"Saved test CSV to {test_output}")


if __name__ == "__main__":
    main()
