# Experimental Replacement Models

Vital now uses two small transfer-learned TensorFlow Lite models generated through the repo's `training/` scaffold.

These models are experimental replacements. They are smaller and easier to reason about than the original bundled models, but they are not clinically validated.

## Skin Triage Model

- File: `app/src/main/ml/skin_triage.tflite`
- Architecture: MobileNetV3Small minimalistic transfer-learning head
- Dataset: public ISIC Archive metadata/images sampled through the public API
- Image source used for this run: 256 px ISIC thumbnails
- Labels:
  - `lower_signal`: ISIC `diagnosis_1 = Benign`
  - `referral`: ISIC `diagnosis_1 = Malignant`
- Training split:
  - Train: 224 images
  - Validation: 48 images
  - Test: 46 images
- TFLite export used in app: dynamic-range quantized float-input model
- Model size: about 535 KB

Held-out test metrics:

- AUROC: 0.819
- Referral threshold selected by script: 0.405
- Lower-signal precision: 0.882
- Lower-signal recall: 0.652
- Referral precision: 0.724
- Referral recall: 0.913
- Confusion matrix:
  - Actual lower-signal: 15 lower-signal, 8 referral
  - Actual referral: 2 lower-signal, 21 referral

Interpretation:

This model prioritizes referral sensitivity on a tiny test set. It is suitable only as an experimental app model and should be retrained with a larger, modality-matched dataset before any serious claim.

## Eye Disease Model

- File: `app/src/main/ml/eyescorer.tflite`
- Architecture: MobileNetV3Small minimalistic transfer-learning head
- Dataset: Kaggle `gunavenkatdoddi/eye-diseases-classification`
- Labels:
  - `cataract`
  - `diabetic_retinopathy`
  - `glaucoma`
  - `normal`
- Training split:
  - Train: 2,953 images
  - Validation: 633 images
  - Test: 631 images
- TFLite export used in app: dynamic-range quantized float-input model
- Model size: about 536 KB

Held-out test metrics:

- Cataract precision/recall/F1: 0.834 / 0.877 / 0.855
- Diabetic retinopathy precision/recall/F1: 0.967 / 0.726 / 0.829
- Glaucoma precision/recall/F1: 0.755 / 0.795 / 0.774
- Normal precision/recall/F1: 0.694 / 0.801 / 0.744
- Confusion matrix by class order `[cataract, diabetic_retinopathy, glaucoma, normal]`:
  - Cataract: 136, 0, 14, 5
  - Diabetic retinopathy: 2, 119, 5, 38
  - Glaucoma: 16, 1, 120, 14
  - Normal: 9, 3, 20, 129

Interpretation:

This model is much smaller than the original eye model and has reasonable experimental test performance on its source dataset. It should still be treated as a dataset-specific model, not a medically validated classifier.

## App Scoring

The Android app maps model outputs into the existing 0-10 score presentation:

- Skin: referral probability is scaled so the selected 0.405 referral threshold maps to score 5.
- Eye: weighted disease probability score using cataract = 7, diabetic retinopathy = 9, glaucoma = 8, normal = 0.

These weights keep the app's existing lower/borderline/elevated result flow intact while reflecting the new model outputs.
