# Experimental Replacement Models

Vital now uses two small transfer-learned TensorFlow Lite models generated through the repo's `training/` scaffold. Both models share the same binary referral architecture: a single sigmoid output trained against a `lower_signal` / `referral` manifest, scored in the app against a referral-friendly threshold selected by the training script.

These models are experimental replacements. They are smaller and easier to reason about than the original bundled models, but they are not clinically validated.

## Skin Triage Model

- File: `app/src/main/ml/skin_triage.tflite`
- Architecture: MobileNetV3Small minimalistic transfer-learning head
- Dataset target: Kaggle `ismailpromus/skin-diseases-image-dataset`
- Manifest builder: `training/build_skin_manifest.py`
- Labels:
  - `lower_signal`: class folders without referral keywords
  - `referral`: class folders containing melanoma, basal, carcinoma, or BCC
- Training sample:
  - Dataset downloaded with KaggleHub from version 1
  - Capped at 1,500 images per diagnosis and 3,000 images per binary label
  - Total rows: 6,000
- Training split:
  - Train: 4,218 images
  - Validation: 865 images
  - Test: 917 images
- TFLite export used in app: dynamic-range quantized float-input model
- Model size: about 535 KB

Held-out test metrics:

- AUROC: 0.922
- Referral threshold selected by script: 0.715
- Lower-signal precision: 0.902
- Lower-signal recall: 0.824
- Referral precision: 0.837
- Referral recall: 0.910
- Confusion matrix:
  - Actual lower-signal: 378 lower-signal, 81 referral
  - Actual referral: 41 lower-signal, 417 referral

Interpretation:

This model uses the same MobileNetV3Small transfer-learning workflow as the eye model and prioritizes referral recall at the selected threshold. It remains an experimental dataset-specific triage model, not a clinically validated diagnostic model.

## Eye Disease Model

- File: `app/src/main/ml/eyescorer.tflite`
- Architecture: MobileNetV3Small minimalistic transfer-learning head (binary referral head, same as the skin model)
- Dataset target: Kaggle `gunavenkatdoddi/eye-diseases-classification`
- Manifest builder: `training/build_eye_manifest.py`
- Labels:
  - `lower_signal`: `normal` class folder
  - `referral`: `cataract`, `diabetic_retinopathy`, and `glaucoma` class folders
- Training sample:
  - Capped at 1,100 images per diagnosis and 3,300 images per binary label
  - Total rows: 4,217
- Training split:
  - Train: 2,948 images
  - Validation: 613 images
  - Test: 656 images
- TFLite export used in app: dynamic-range quantized float-input model
- Model size: about 536 KB

Held-out test metrics:

- AUROC: 0.937
- Referral threshold selected by script: 0.515
- Lower-signal precision: 0.765
- Lower-signal recall: 0.777
- Referral precision: 0.913
- Referral recall: 0.907
- Confusion matrix:
  - Actual lower-signal: 143 lower-signal, 41 referral
  - Actual referral: 44 lower-signal, 428 referral

Interpretation:

This model now uses the exact same binary referral architecture and training script as the skin model: any of the three detected eye pathologies in the source dataset (cataract, diabetic retinopathy, glaucoma) is collapsed into a single `referral` class against `normal` as `lower_signal`. It remains an experimental dataset-specific triage model, not a clinically validated diagnostic model, and it no longer reports per-disease categorization — disease-specific risk categorization was intentionally replaced by a referral-oriented binary signal to match the skin model's triage framing.

## App Scoring

Both analysis paths now share one scoring function. The Android app scales each model's referral probability so its own selected threshold maps to score 5 on the existing 0-10 presentation:

- Skin: referral probability is scaled so the selected 0.715 referral threshold maps to score 5.
- Eye: referral probability is scaled so the selected 0.515 referral threshold maps to score 5.

This keeps the app's existing lower/borderline/elevated result flow intact while giving both models an identical referral-oriented contract.
