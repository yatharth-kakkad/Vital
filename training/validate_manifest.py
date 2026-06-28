import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image


REQUIRED_COLUMNS = {"image_path", "label", "split"}
VALID_SPLITS = {"train", "val", "test"}


def resolve_path(path_value: str, dataset_root: Optional[Path]) -> Path:
    path = Path(path_value)
    if path.is_absolute() or dataset_root is None:
        return path
    return dataset_root / path


def validate_manifest(manifest_path: Path, dataset_root: Optional[Path], check_images: bool) -> None:
    frame = pd.read_csv(manifest_path)
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

    unknown_splits = set(frame["split"].astype(str).str.lower()) - VALID_SPLITS
    if unknown_splits:
        raise ValueError(f"Unknown split values: {sorted(unknown_splits)}")

    split_counts = frame["split"].astype(str).str.lower().value_counts().to_dict()
    for split in sorted(VALID_SPLITS):
        if split_counts.get(split, 0) == 0:
            raise ValueError(f"Split '{split}' has no rows")

    labels = sorted(frame["label"].astype(str).unique())
    if len(labels) < 2:
        raise ValueError("At least two labels are required")

    print("Rows:", len(frame))
    print("Labels:", labels)
    print("Split counts:", split_counts)
    print("Label by split:")
    print(pd.crosstab(frame["split"], frame["label"]))

    if not check_images:
        return

    missing = []
    unreadable = []
    for path_value in frame["image_path"]:
        path = resolve_path(str(path_value), dataset_root)
        if not path.exists():
            missing.append(str(path))
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            unreadable.append(str(path))

    if missing:
        raise FileNotFoundError(f"{len(missing)} images are missing. First missing: {missing[0]}")
    if unreadable:
        raise ValueError(f"{len(unreadable)} images are unreadable. First unreadable: {unreadable[0]}")

    print("Image check: passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Vital training manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--skip-image-check", action="store_true")
    args = parser.parse_args()

    validate_manifest(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        check_images=not args.skip_image_check,
    )


if __name__ == "__main__":
    main()
