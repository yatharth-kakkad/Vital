# Training Pipeline

This directory contains a reproducible baseline pipeline for training better lightweight models for Vital without changing the current Android app behavior.

The pipeline is intentionally conservative:

- It uses transfer learning instead of training from scratch.
- It targets small mobile backbones suitable for old Android devices.
- It separates train, validation, and test splits by manifest.
- It reports triage-oriented metrics instead of only accuracy.
- It exports float and quantized TensorFlow Lite models.

The current app still uses the checked-in models under `app/src/main/ml/`. New models should only replace those files after validation results are reviewed and documented.

## Dataset Format

Create a CSV manifest with one row per image:

```csv
image_path,label,split,source,skin_tone,device
/absolute/path/to/image1.jpg,benign,train,isic,,unknown
/absolute/path/to/image2.jpg,referral,test,isic,,unknown
```

Required columns:

- `image_path`: absolute or dataset-root-relative image path.
- `label`: class label.
- `split`: one of `train`, `val`, or `test`.

Optional columns such as `source`, `skin_tone`, `device`, `site`, or `country` are preserved for future subgroup checks.

For the app narrative, a binary triage dataset is the simplest responsible starting point:

- `lower_signal`
- `referral`

Multi-class training is supported, but the app copy should remain referral-oriented unless thresholds are clinically validated.

## Quick Start

```bash
cd training
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python validate_manifest.py --manifest /path/to/manifest.csv --dataset-root /path/to/images
python train_mobile_triage.py \
  --manifest /path/to/manifest.csv \
  --dataset-root /path/to/images \
  --output-dir runs/skin_baseline \
  --positive-label referral \
  --epochs 20 \
  --fine-tune-epochs 8
```

Outputs:

- `best_model.keras`: best validation checkpoint.
- `metrics.json`: test metrics and threshold summary.
- `confusion_matrix.csv`: test confusion matrix at the chosen threshold.
- `label_map.json`: class ordering.
- `model_float32.tflite`: float TFLite export.
- `model_dynamic_range.tflite`: smaller dynamic-range quantized export.

## Replacing App Models

Do not overwrite app models just because a training run completes. Before replacement:

1. Confirm test performance on held-out data.
2. Check sensitivity at a referral-friendly threshold.
3. Review subgroup behavior where metadata exists.
4. Fill out `docs/MODEL_CARD.md`.
5. Add the new `.tflite` model and update Android inference code if input/output shapes changed.

## Why Transfer Learning

Training from scratch is unlikely to work well without a large, clinically representative dataset. Transfer learning with a small backbone is the practical path for a student prototype: it is faster, easier to reproduce, and more aligned with deployment on decade-old hardware.
