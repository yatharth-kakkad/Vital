# Model Card Template

This model card should be completed before replacing any model under `app/src/main/ml/`.

The current experimental replacements are documented in `docs/EXPERIMENTAL_MODEL_RESULTS.md`.

## Model Details

- Model name:
- Version:
- Date:
- Intended app target:
- Input shape:
- Output shape:
- TFLite file:
- Quantization:

## Intended Use

Vital models are intended for offline triage support in low-resource environments. They should help decide whether a case needs repeat imaging or referral to a qualified clinician.

They are not intended for autonomous diagnosis, emergency decision-making, or use without clinical oversight.

## Dataset

- Dataset sources:
- Label definitions:
- Number of images:
- Train/validation/test split strategy:
- Known geography/device/site limitations:
- Known skin tone or demographic coverage:

## Training

- Training script:
- Backbone:
- Pretraining source:
- Image size:
- Augmentation:
- Optimizer and learning rates:
- Epochs:
- Early stopping criteria:

## Evaluation

Report held-out test metrics here.

- AUROC:
- Sensitivity/recall:
- Specificity:
- Precision:
- F1:
- Chosen referral threshold:
- Confusion matrix:
- Calibration notes:

## Subgroup Checks

Where metadata is available, report performance by:

- Skin tone or visible phenotype
- Device/camera type
- Dataset source/site
- Age group
- Sex/gender
- Geography

## Limitations

- Dataset bias:
- Failure modes:
- Conditions not represented:
- Image quality concerns:
- Expected behavior on older devices:

## Release Decision

- Approved for replacing app model: no
- Reviewer:
- Date:
- Reasoning:
