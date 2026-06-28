import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


IMAGE_SIZE = 224
AUTOTUNE = tf.data.AUTOTUNE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def resolve_path(path_value: str, dataset_root: Optional[Path]) -> str:
    path = Path(path_value)
    if path.is_absolute() or dataset_root is None:
        return str(path)
    return str(dataset_root / path)


def load_manifest(
    manifest_path: Path,
    dataset_root: Optional[Path],
    positive_label: Optional[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = pd.read_csv(manifest_path)
    frame["split"] = frame["split"].astype(str).str.lower()
    frame["label"] = frame["label"].astype(str)
    frame["resolved_path"] = frame["image_path"].astype(str).map(lambda value: resolve_path(value, dataset_root))

    labels = sorted(frame["label"].unique())
    if positive_label and len(labels) == 2:
        if positive_label not in labels:
            raise ValueError(f"Positive label '{positive_label}' is not present in the manifest")
        negative_label = next(label for label in labels if label != positive_label)
        label_map = {negative_label: 0, positive_label: 1}
    else:
        label_map = {label: index for index, label in enumerate(labels)}
    frame["label_index"] = frame["label"].map(label_map).astype(int)
    return frame, label_map


def decode_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32)
    return image, label


def augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label


def make_dataset(frame: pd.DataFrame, batch_size: int, training: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (frame["resolved_path"].to_numpy(), frame["label_index"].to_numpy())
    )
    if training:
        dataset = dataset.shuffle(buffer_size=len(frame), reshuffle_each_iteration=True)
    dataset = dataset.map(decode_image, num_parallel_calls=AUTOTUNE)
    if training:
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def build_model(num_classes: int, dropout: float) -> tuple[tf.keras.Model, tf.keras.Model]:
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")
    preprocessed = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    backbone = tf.keras.applications.MobileNetV3Small(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        pooling="avg",
        minimalistic=True,
    )
    backbone.trainable = False

    x = backbone(preprocessed)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(
        1 if num_classes == 2 else num_classes,
        activation="sigmoid" if num_classes == 2 else "softmax",
        name="risk",
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs), backbone


def compile_model(model: tf.keras.Model, num_classes: int, learning_rate: float) -> None:
    if num_classes == 2:
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auroc"),
            tf.keras.metrics.AUC(curve="PR", name="auprc"),
        ]
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray, min_sensitivity: float) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_specificity = -1.0
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        if sensitivity >= min_sensitivity and specificity > best_specificity:
            best_threshold = float(threshold)
            best_specificity = specificity
    return best_threshold


def evaluate_binary(model: tf.keras.Model, dataset: tf.data.Dataset, output_dir: Path, min_sensitivity: float) -> dict:
    y_true = np.concatenate([labels.numpy() for _, labels in dataset])
    y_score = model.predict(dataset).reshape(-1)
    threshold = choose_threshold(y_true, y_score, min_sensitivity)
    y_pred = (y_score >= threshold).astype(int)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    metrics = {
        "threshold": threshold,
        "auroc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None,
        "precision_by_class": precision.tolist(),
        "recall_by_class": recall.tolist(),
        "f1_by_class": f1.tolist(),
        "confusion_matrix": matrix.tolist(),
    }
    pd.DataFrame(matrix, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    return metrics


def evaluate_multiclass(model: tf.keras.Model, dataset: tf.data.Dataset, output_dir: Path) -> dict:
    y_true = np.concatenate([labels.numpy() for _, labels in dataset])
    y_prob = model.predict(dataset)
    y_pred = y_prob.argmax(axis=1)
    labels = sorted(np.unique(y_true).tolist())
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    pd.DataFrame(matrix).to_csv(output_dir / "confusion_matrix.csv", index=False)
    return {
        "precision_by_class": precision.tolist(),
        "recall_by_class": recall.tolist(),
        "f1_by_class": f1.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def representative_dataset(frame: pd.DataFrame, max_samples: int):
    sample = frame.head(max_samples)
    for path in sample["resolved_path"]:
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
        image = tf.cast(image, tf.float32)
        yield [tf.expand_dims(image, axis=0)]


def export_tflite(model: tf.keras.Model, output_dir: Path, train_frame: pd.DataFrame) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    (output_dir / "model_float32.tflite").write_bytes(converter.convert())

    quantized_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    quantized_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    (output_dir / "model_dynamic_range.tflite").write_bytes(quantized_converter.convert())

    int8_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    int8_converter.representative_dataset = lambda: representative_dataset(train_frame, max_samples=100)
    int8_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    int8_converter.inference_input_type = tf.uint8
    int8_converter.inference_output_type = tf.uint8
    (output_dir / "model_int8.tflite").write_bytes(int8_converter.convert())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight Vital triage model.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--fine-tune-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--min-sensitivity", type=float, default=0.9)
    parser.add_argument("--positive-label", default="referral")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, label_map = load_manifest(args.manifest, args.dataset_root, args.positive_label)
    train_frame = frame[frame["split"] == "train"].copy()
    val_frame = frame[frame["split"] == "val"].copy()
    test_frame = frame[frame["split"] == "test"].copy()

    num_classes = len(label_map)
    train_ds = make_dataset(train_frame, args.batch_size, training=True)
    val_ds = make_dataset(val_frame, args.batch_size, training=False)
    test_ds = make_dataset(test_frame, args.batch_size, training=False)

    model, backbone = build_model(num_classes=num_classes, dropout=args.dropout)
    compile_model(model, num_classes=num_classes, learning_rate=args.learning_rate)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.output_dir / "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    backbone.trainable = True
    for layer in backbone.layers[:-20]:
        layer.trainable = False
    compile_model(model, num_classes=num_classes, learning_rate=args.fine_tune_learning_rate)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks,
    )

    model.save(args.output_dir / "final_model.keras")
    (args.output_dir / "label_map.json").write_text(json.dumps(label_map, indent=2) + "\n")

    if num_classes == 2:
        metrics = evaluate_binary(model, test_ds, args.output_dir, args.min_sensitivity)
    else:
        metrics = evaluate_multiclass(model, test_ds, args.output_dir)
    metrics["label_map"] = label_map
    metrics["image_size"] = IMAGE_SIZE
    metrics["backbone"] = "MobileNetV3Small minimalistic"
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    export_tflite(model, args.output_dir, train_frame)


if __name__ == "__main__":
    main()
