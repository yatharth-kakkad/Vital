# Dataset Selection for Replacement Models

This note records why Vital does not currently include newly trained replacement models, even though the repo now has a training pipeline.

## Summary

Training technically better `.tflite` files is plausible for the skin path, but not as a quick, responsible one-shot replacement for both skin and eye models.

The blocker is not the training code. The blocker is dataset fit:

- Credible skin datasets exist, especially ISIC challenge datasets.
- Most credible skin datasets are dermoscopy or clinical lesion crops, not ordinary phone-camera images.
- Credible public eye datasets are usually retinal/fundus datasets, not visible-eye phone photos.
- A retinal model would not be a drop-in replacement for a normal camera eye-screening flow.

Replacing the app models should wait until the dataset matches the capture modality and intended deployment environment.

## Skin Data Candidates

### ISIC Challenge Datasets

The ISIC challenge archive provides multiple skin lesion datasets, including binary malignant-status tasks and larger diagnosis datasets. These are credible starting points for a skin triage model.

Best fit for Vital's low-resource narrative:

- ISIC SLICE-3D permissive subset: lesion crops from 3D total body photography, CC-BY, with metadata.
- ISIC 2016 Task 3: smaller binary malignant-status dataset, useful for a fast baseline.

Limitations:

- These datasets do not perfectly represent low-cost phone-camera capture.
- Dermoscopy-trained models may fail on ordinary clinical photos.
- Subgroup coverage must be checked before any field-facing claim.

## Eye Data Candidates

Credible public eye datasets tend to be for diabetic retinopathy or retinal disease classification from fundus images. Those images require specialized retinal cameras and are not the same as the app's visible-eye camera workflow.

That means they are poor drop-in replacements unless the app is explicitly repositioned as a fundus-image tool.

Before training an eye replacement model, choose one of these directions:

1. Keep the current visible-eye workflow and collect/curate visible-eye images with labels.
2. Reposition the app's eye path for fundus/retinal screening and update UI copy, capture instructions, and inference assumptions.
3. Remove the eye model from replacement scope until a suitable dataset is available.

## Replacement Acceptance Criteria

A replacement model can be considered only when all of the following are true:

- The dataset image type matches the app capture flow.
- Train, validation, and test splits are separated before model selection.
- Test metrics are reported in `docs/MODEL_CARD.md`.
- The positive/referral class is explicit.
- Sensitivity is prioritized at the selected referral threshold.
- Subgroup checks are reported where metadata exists.
- The exported TFLite model is smaller or faster than the existing model path.
- Android inference code is updated and verified if tensor shapes or labels change.

## Practical Next Step

Run a skin-only baseline first using the training scaffold and ISIC data. Treat the resulting model as an experiment, not a replacement, until the model card shows credible held-out performance.

For eye screening, do not train from fundus datasets unless the app is changed to require fundus images.
