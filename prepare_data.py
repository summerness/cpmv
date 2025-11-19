import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.model_selection import StratifiedKFold

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
MASK_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy", ".npz")


def list_images(directory: Path) -> List[Path]:
    files: List[Path] = []
    for ext in IMAGE_EXTS:
        files.extend(directory.rglob(f"*{ext}"))
    return sorted([p for p in files if p.is_file()])


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
        rel_parts = image_path.relative_to(image_dir).parts
        category = rel_parts[0] if len(rel_parts) > 1 else "root"
        mask_path = None
        if category.lower() != "authentic":
            mask_path = find_mask(mask_dir, stem)
            if mask_path is None:
                if allow_missing_mask:
                    continue
                raise FileNotFoundError(f"Mask for {image_path} not found in {mask_dir}")
        rows.append({
            "image_path": str(image_path),
            "mask_path": str(mask_path) if mask_path is not None else "",
            "source": source_label,
            "category": category,
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
    parser.add_argument("--dataset-root", type=str, default=None, help="Optional base directory (e.g., /kaggle/input/... ).")
    parser.add_argument("--train-images", type=str, default=None)
    parser.add_argument("--train-masks", type=str, default=None)
    parser.add_argument("--train-output", type=str, default="data/train.csv")
    parser.add_argument("--supp-images", type=str, default=None)
    parser.add_argument("--supp-masks", type=str, default=None)
    parser.add_argument("--include-supp", action="store_true")
    parser.add_argument("--test-images", type=str, default=None)
    parser.add_argument("--test-output", type=str, default="data/test.csv")
    parser.add_argument("--sample-submission", type=str, default=None)
    parser.add_argument("--allow-missing-mask", action="store_true")
    parser.add_argument("--num-folds", type=int, default=0, help="Number of folds to assign (0 to disable).")
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true", help="Disable final row shuffling before saving CSV.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root) if args.dataset_root else None

    def resolve_path(provided: Optional[str], default_subdir: str, must_exist: bool = True) -> Optional[Path]:
        if provided:
            path = Path(provided)
        elif dataset_root:
            path = dataset_root / default_subdir
        else:
            return None
        if must_exist and not path.exists():
            raise FileNotFoundError(f"Expected path {path} does not exist")
        return path

    train_dir = resolve_path(args.train_images, "train_images")
    train_mask_dir = resolve_path(args.train_masks, "train_masks")
    if train_dir is None or train_mask_dir is None:
        raise ValueError("Please provide --train-images/--train-masks or --dataset-root with train_images/train_masks.")

    train_output = Path(args.train_output)
    train_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building train CSV from {train_dir} and {train_mask_dir}")
    train_df = build_train_df(train_dir, train_mask_dir, source_label="primary", allow_missing_mask=args.allow_missing_mask)

    supp_images = resolve_path(args.supp_images, "supplemental_images", must_exist=False)
    supp_masks = resolve_path(args.supp_masks, "supplemental_masks", must_exist=False)
    if args.include_supp and supp_images and supp_masks:
        if not supp_images.exists() or not supp_masks.exists():
            raise FileNotFoundError("Supplemental directories not found.")
        supp_df = build_train_df(
            supp_images,
            supp_masks,
            source_label="supplemental",
            allow_missing_mask=args.allow_missing_mask,
        )
        combined = pd.concat([train_df, supp_df], ignore_index=True)
    else:
        combined = train_df

    if args.num_folds and args.num_folds > 1:
        labels = (combined["mask_path"].astype(str).str.len() > 0).astype(int)
        combined["fold"] = -1
        if labels.nunique() > 1:
            skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.fold_seed)
            for fold_id, (_, val_idx) in enumerate(skf.split(combined, labels)):
                combined.loc[val_idx, "fold"] = fold_id
        else:
            combined["fold"] = (range(len(combined)) % args.num_folds)
        print(f"Assigned stratified folds with n_splits={args.num_folds}")
    if not args.no_shuffle:
        forged = combined[combined["category"].str.lower() != "authentic"]
        if not forged.empty:
            first_forged = forged.sample(n=1, random_state=args.fold_seed)
            combined = pd.concat([first_forged, combined.drop(first_forged.index)], ignore_index=True)
        combined = combined.sample(frac=1, random_state=args.fold_seed).reset_index(drop=True)
    combined.to_csv(train_output, index=False)
    print(f"Saved {len(combined)} rows to {train_output}")

    test_dir = resolve_path(args.test_images, "test_images", must_exist=False)
    if test_dir and test_dir.exists():
        test_output = Path(args.test_output)
        test_output.parent.mkdir(parents=True, exist_ok=True)
        sample_sub = resolve_path(args.sample_submission, "sample_submission.csv", must_exist=False)
        build_test_csv(test_dir, test_output, sample_sub)
        print(f"Saved test CSV to {test_output}")


if __name__ == "__main__":
    main()
