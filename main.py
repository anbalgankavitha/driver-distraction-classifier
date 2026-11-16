import os
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Project configuration
CLASSES = [
    "safe_driving",
    "texting",
    "talking_on_phone",
    "drinking",
    "drowsy_or_distracted",
]

IMG_SIZE = (128, 128)
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
VAL_DIR = BASE_DIR / "dataset" / "val"
MODEL_PATH = BASE_DIR / "driver_distraction_model.joblib"


def prepare_sample_dataset():
    """
    Create a small artificial dataset so the project can run immediately.

    IMPORTANT: These generated images are only for demonstrating the
    machine-learning pipeline. They are not real driver-behaviour images
    and must not be used to evaluate a real-world safety system.
    """
    rng = np.random.default_rng(42)

    for split_dir, image_count in [(TRAIN_DIR, 30), (VAL_DIR, 10)]:
        for class_index, class_name in enumerate(CLASSES):
            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for image_index in range(image_count):
                image = np.zeros((128, 128, 3), dtype=np.uint8)

                base_color = (
                    (class_index * 50) % 255,
                    (class_index * 80 + 40) % 255,
                    (class_index * 110 + 90) % 255,
                )
                image[:] = base_color

                cv2.circle(
                    image,
                    (64, 40 + class_index * 8),
                    18,
                    (255, 255, 255),
                    -1,
                )

                noise = rng.integers(
                    0, 20, size=image.shape, dtype=np.uint8
                )
                image = cv2.add(image, noise)

                output_path = class_dir / f"{class_name}_{image_index}.jpg"
                if not cv2.imwrite(str(output_path), image):
                    raise RuntimeError(f"Could not save image: {output_path}")


def load_images_and_features(directory):
    """Load class images and convert each image into a feature vector."""
    features = []
    labels = []

    for class_index, class_name in enumerate(CLASSES):
        class_dir = directory / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class folder: {class_dir}. "
                "Run prepare_sample_dataset() first."
            )

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: skipping unreadable image: {image_path}")
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, IMG_SIZE).astype("float32") / 255.0

            histogram = cv2.calcHist(
                [rgb],
                [0, 1, 2],
                None,
                [8, 8, 8],
                [0, 256, 0, 256, 0, 256],
            ).flatten()

            image_features = np.concatenate(
                [resized.flatten()[::16], histogram]
            )
            features.append(image_features)
            labels.append(class_index)

    if not features:
        raise ValueError(f"No readable images found in {directory}")

    return np.array(features), np.array(labels)


def main():
    # Generate sample images for a first-time demo run.
    prepare_sample_dataset()

    X_train, y_train = load_images_and_features(TRAIN_DIR)
    X_val, y_val = load_images_and_features(VAL_DIR)

    print(
        f"Found {len(X_train)} training images belonging to "
        f"{len(CLASSES)} classes."
    )
    print(
        f"Found {len(X_val)} validation images belonging to "
        f"{len(CLASSES)} classes."
    )

    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_val)

    print(
        f"\nValidation Accuracy: "
        f"{accuracy_score(y_val, predictions) * 100:.2f}%\n"
    )
    print(
        classification_report(
            y_val,
            predictions,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            zero_division=0,
        )
    )

    # Save the trained model and the class-name mapping.
    joblib.dump(
        {"model": classifier, "classes": CLASSES},
        MODEL_PATH,
    )
    print(f"Model saved to: {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
