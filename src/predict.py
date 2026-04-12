"""
Run inference on a single MRI image.

Usage:
    python src/predict.py --image path/to/mri.jpg
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import keras

from config import MODEL_PATH, CLASSES, DISPLAY_NAMES
from src.utils import preprocess_image


def predict(image_path: str) -> None:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python src/train.py` first."
        )

    model = keras.models.load_model(MODEL_PATH)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    tensor = preprocess_image(img)
    scores = model.predict(tensor)[0]

    predicted_idx = scores.argmax()
    predicted_class = CLASSES[predicted_idx]

    print(f"Prediction: {DISPLAY_NAMES[predicted_class]}  ({scores[predicted_idx]*100:.1f}% confidence)\n")
    print("Per-class probabilities:")
    for cls, score in zip(CLASSES, scores):
        bar = "#" * int(score * 40)
        print(f"  {DISPLAY_NAMES[cls]:<14} {score*100:5.1f}%  {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a brain tumor MRI image.")
    parser.add_argument("--image", required=True, help="Path to the MRI image file")
    args = parser.parse_args()
    predict(args.image)
