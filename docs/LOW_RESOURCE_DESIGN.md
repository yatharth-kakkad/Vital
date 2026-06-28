# Low-Resource Design Notes

Vital is framed as a field triage prototype for settings where bandwidth, device age, and clinical access are constrained.

## Design Goals

- **Offline first:** inference runs with bundled TensorFlow Lite models and does not require a server.
- **Older hardware:** the app supports Android 5.0+ and keeps input images to a fixed 224 x 224 crop.
- **Predictable memory:** model preprocessing resizes images before inference, avoiding large arbitrary camera bitmaps in the model path.
- **Responsive UI:** inference runs on a background dispatcher so slower phones still show progress instead of appearing frozen.
- **Careful language:** results are framed as triage/referral prompts, not diagnoses.

## Model Pipeline

Both analysis paths use one 224 x 224 crop, one bundled TensorFlow Lite model with an identical binary referral contract, and the same score mapping in the app:

- Skin: MobileNetV3Small referral model trained from a balanced subset of `ismailpromus/skin-diseases-image-dataset`; referral probability is scaled against the documented referral threshold.
- Eye: MobileNetV3Small referral model trained the same way from `gunavenkatdoddi/eye-diseases-classification`, collapsing cataract/diabetic retinopathy/glaucoma into `referral` and normal into `lower_signal`; referral probability is scaled against its own documented referral threshold.

The model instances are closed in `finally` blocks so repeated screening sessions do not leak native resources.

## Deployment Tradeoffs

Bundling models makes the APK larger, but it also makes the app independent of network availability. For production deployment, the next step would be evaluating quantized model variants and Git LFS or release assets for model storage.

## Training and Model Replacement

The `training/` directory provides a transfer-learning pipeline for future model improvement. It does not replace the current app models automatically. Any replacement model should include:

- a documented dataset manifest,
- held-out test metrics,
- referral-threshold selection,
- subgroup checks where metadata exists,
- a short release note in `docs/EXPERIMENTAL_MODEL_RESULTS.md`,
- and an Android inference update if tensor shapes or label order change.

For this app's low-resource narrative, small transfer-learned and quantized models are a better fit than training large models from scratch.

The current experimental model metrics live in `docs/EXPERIMENTAL_MODEL_RESULTS.md`.

## Safety Boundary

The app should be described as disease identification support or triage assistance, not automated diagnosis. Any elevated or uncertain signal should route the patient to a qualified clinician.
