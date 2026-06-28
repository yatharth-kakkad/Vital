import argparse
import hashlib
import random
from pathlib import Path

import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REFERRAL_KEYWORDS = {
    "akiec",
    "basal",
    "bcc",
    "carcinoma",
    "melanoma",
    "malignant",
    "scc",
    "squamous",
}


def split_for(path: Path, seed: int) -> str:
    bucket = int(hashlib.sha1(f"{seed}:{path}".encode()).hexdigest(), 16) % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


def label_for(class_name: str) -> str:
    normalized = class_name.lower().replace("-", "_").replace(" ", "_")
    return "referral" if any(keyword in normalized for keyword in REFERRAL_KEYWORDS) else "lower_signal"


def images_under(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Vital binary skin-triage manifest from the Kaggle skin disease image dataset."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-per-diagnosis", type=int)
    parser.add_argument("--max-per-label", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    random.seed(args.seed)
    class_dirs = sorted(path for path in args.dataset_root.rglob("*") if path.is_dir() and images_under(path))
    for class_dir in class_dirs:
        class_name = class_dir.name
        images = images_under(class_dir)
        if args.max_per_diagnosis:
            random.shuffle(images)
            images = sorted(images[: args.max_per_diagnosis])
        for image_path in images:
            rows.append(
                {
                    "image_path": str(image_path.relative_to(args.dataset_root)),
                    "label": label_for(class_name),
                    "split": split_for(image_path.relative_to(args.dataset_root), args.seed),
                    "source": "kaggle:ismailpromus/skin-diseases-image-dataset",
                    "diagnosis": class_name,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No images found under {args.dataset_root}")
    if args.max_per_label:
        frame = (
            frame.sample(frac=1, random_state=args.seed)
            .groupby("label", group_keys=False)
            .head(args.max_per_label)
            .sort_values(["label", "diagnosis", "image_path"])
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print("Rows:", len(frame))
    print(pd.crosstab(frame["split"], frame["label"]))


if __name__ == "__main__":
    main()
