# Vital

Vital is a lightweight Android prototype for skin and eye disease triage in low-resource settings. It is designed to run local TensorFlow Lite inference on older Android hardware so a clinic or outreach worker can screen images without a network connection, cloud account, or modern device.

> Vital is an educational prototype, not a medical device. It produces a triage signal and should never be treated as a diagnosis or as a replacement for professional care.

## Why It Exists

Many health tools quietly assume reliable broadband, current phones, and easy access to specialists. Vital takes the opposite constraint set: impoverished or remote regions, older donated devices, sensitive medical images, and intermittent connectivity. The app keeps inference local and uses fixed-size image crops so runtime and memory use stay predictable.

## Highlights

- Fully local inference; captured images are not uploaded.
- Fixed 224 x 224 crop flow to match model input and control memory use.
- Separate skin and eye analysis paths backed by bundled `.tflite` models, both using the same binary referral architecture.
- Background inference so the UI stays responsive on slower devices.
- Simple referral-oriented language instead of overclaiming a diagnosis or per-disease categorization.
- Minimum SDK 21, targeting Android 5.0+ devices from roughly the last decade.

## Field Workflow

1. Take a close, well-lit photo of the affected skin or eye region.
2. Crop the relevant region to the guided square.
3. Run the skin or eye model locally.
4. Use the result as a triage prompt: lower signal, repeat/review, or high-priority referral.

## Tech Stack

- Kotlin
- Jetpack Compose and Material 3
- Android Navigation Compose
- TensorFlow Lite Support
- CanHub Android Image Cropper

## Getting Started

Requirements:

- Android Studio with a JDK 17 runtime
- Android SDK 36 or newer installed

Run from Android Studio or from the command line:

```bash
./gradlew :app:assembleDebug
```

The app requests camera access at runtime. Captured images are written to the app cache and passed through the cropper before inference.

## Project Structure

- `app/src/main/java/com/example/detector/MainActivity.kt` contains the Compose UI and model inference orchestration.
- `app/src/main/ml/` contains the TensorFlow Lite models.
- `app/src/main/res/xml/file_paths.xml` defines cache access for camera captures.
- `docs/LOW_RESOURCE_DESIGN.md` documents the low-resource deployment decisions.
- `docs/EXPERIMENTAL_MODEL_RESULTS.md` documents the current scaffold-generated model replacements.
- `training/` contains a reproducible transfer-learning pipeline for future model improvements.

## Current Limitations

- The included models are bundled directly in the app, which makes the repository and APK large.
- The score thresholds are simple and should be calibrated against validated datasets before any real-world use.
- The app does not yet include multilingual copy, offline training material, or patient record export.
- The included replacement models were generated with the training scaffold, but they are experimental and not clinically validated.
