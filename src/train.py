"""
Train a MobileNet-based brain tumor classifier.

Usage:
    python src/train.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score,
)
import keras
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense, Dropout
from keras.models import Sequential

from config import (
    TRAIN_DIR, MODEL_PATH, MODEL_DIR,
    IMAGE_SIZE, EPOCHS, TEST_SPLIT, RANDOM_STATE,
    DROPOUT_RATE, DENSE_UNITS,
)
from src.utils import load_images_from_dir


def build_model() -> Sequential:
    base = MobileNet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, pooling="max")
    model = Sequential([
        base,
        Dropout(DROPOUT_RATE),
        Dense(DENSE_UNITS, activation="relu"),
        Dropout(DROPOUT_RATE),
        Dense(4, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate(model: Sequential, X_test: np.ndarray, y_test: np.ndarray) -> None:
    y_pred = model.predict(X_test).argmax(axis=1).tolist()
    y_true = y_test.tolist()
    print(f"Accuracy:     {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision:    {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall:       {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1 Score:     {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Cohen Kappa:  {cohen_kappa_score(y_true, y_pred):.4f}")


def main() -> None:
    print("Loading training data...")
    X, y = load_images_from_dir(TRAIN_DIR)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

    print("\nBuilding model...")
    model = build_model()
    model.summary()

    print("\nTraining...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=32,
    )

    print("\nEvaluation:")
    evaluate(model, X_test, y_test)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
