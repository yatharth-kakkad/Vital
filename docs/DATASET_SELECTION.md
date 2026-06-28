# Dataset Selection for Replacement Models

This note records the dataset-fit caveats for Vital's scaffold-generated replacement models.

## Summary

Training technically better `.tflite` files is plausible, but the resulting files remain experimental unless the dataset matches the app's capture setting and the model passes stronger validation.

The blocker is not the training code. The blocker is dataset fit:

- Credible skin datasets exist, especially ISIC challenge datasets.
- Most credible skin datasets are dermoscopy or clinical lesion crops, not ordinary phone-camera images.
- Credible public eye datasets are usually retinal/fundus datasets, not visible-eye phone photos.
- A retinal model would not be a drop-in replacement for a normal camera eye-screening flow.

Vital now includes scaffold-generated experimental replacements. They should be treated as prototypes, not validated medical models.

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

Retrain the skin model with a larger ISIC sample or a phone-camera clinical image dataset, then compare against the current experimental model in `docs/EXPERIMENTAL_MODEL_RESULTS.md`.

For eye screening, continue using the Kaggle eye-disease dataset lineage unless a more representative visible-eye dataset is available.
