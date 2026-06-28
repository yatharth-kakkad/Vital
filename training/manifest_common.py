import argparse
import hashlib
import random
from pathlib import Path
from typing import Callable, Optional

import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def split_for(path: Path, seed: int) -> str:
    bucket = int(hashlib.sha1(f"{seed}:{path}".encode()).hexdigest(), 16) % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


def images_under(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def build_manifest(
    dataset_root: Path,
    label_for: Callable[[str], str],
    source: str,
    seed: int,
    max_per_diagnosis: Optional[int],
    max_per_label: Optional[int],
) -> pd.DataFrame:
    rows = []
    random.seed(seed)
    class_dirs = sorted(path for path in dataset_root.rglob("*") if path.is_dir() and images_under(path))
    for class_dir in class_dirs:
        images = images_under(class_dir)
        if max_per_diagnosis:
            random.shuffle(images)
            images = sorted(images[:max_per_diagnosis])
        for image_path in images:
            rows.append(
                {
                    "image_path": str(image_path.relative_to(dataset_root)),
                    "label": label_for(class_dir.name),
                    "split": split_for(image_path.relative_to(dataset_root), seed),
                    "source": source,
                    "diagnosis": class_dir.name,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No images found under {dataset_root}")
    if max_per_label:
        frame = (
            frame.sample(frac=1, random_state=seed)
            .groupby("label", group_keys=False)
            .head(max_per_label)
            .sort_values(["label", "diagnosis", "image_path"])
        )
    return frame


def manifest_cli(description: str, label_for: Callable[[str], str], source: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-per-diagnosis", type=int)
    parser.add_argument("--max-per-label", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = build_manifest(
        args.dataset_root, label_for, source, args.seed, args.max_per_diagnosis, args.max_per_label
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print("Rows:", len(frame))
    print(pd.crosstab(frame["split"], frame["label"]))
