# Vital

Vital is an Android prototype for camera-based skin and eye screening. It captures a photo, guides the user through a fixed 224 x 224 crop, and runs bundled TensorFlow Lite models on device to produce a simple risk signal.

> This is an educational prototype and not a medical device. It should not be used as a diagnosis or as a replacement for professional care.

## Highlights

- Jetpack Compose UI with a clean capture, crop, analyze, and result flow.
- On-device TensorFlow Lite inference; no image upload is required.
- Separate skin and eye analysis paths backed by bundled `.tflite` models.
- Camera capture through `FileProvider` and a fixed-aspect cropper for consistent model input.

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

1. Open the project in Android Studio.
2. Let Gradle sync the dependencies.
3. Run the `app` configuration on an Android device or emulator with a camera.

The app requests camera access at runtime. Captured images are written to the app cache and passed through the cropper before inference.

## Project Structure

- `app/src/main/java/com/example/detector/MainActivity.kt` contains the Compose UI and model inference orchestration.
- `app/src/main/ml/` contains the TensorFlow Lite models.
- `app/src/main/res/xml/file_paths.xml` defines cache access for camera captures.

## Notes

The scoring thresholds are intentionally simple so the prototype can communicate a readable signal. Any health concern should be checked by a qualified clinician, regardless of the app output.
