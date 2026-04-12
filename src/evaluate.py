"""
Generate and save evaluation plots for the trained model.

Produces:
  assets/confusion_matrix.png
  assets/roc_curve.png

Usage:
    python src/evaluate.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import itertools

import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from config import ASSETS_DIR, CLASSES, DISPLAY_NAMES, MODEL_PATH, TEST_DIR
from src.utils import load_images_from_dir

DISPLAY = [DISPLAY_NAMES[c] for c in CLASSES]


def plot_confusion_matrix(y_true: list, y_pred: list, save_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DISPLAY)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — MobileNet", fontsize=14, pad=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_roc_curves(y_true: list, y_scores: np.ndarray, save_path: str) -> None:
    n_classes = len(CLASSES)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b"]

    for i, (cls, color) in enumerate(zip(CLASSES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_scores[:, i])
        auc = roc_auc_score(y_bin[:, i], y_scores[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{DISPLAY_NAMES[cls]} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — MobileNet (One-vs-Rest)", fontsize=14, pad=12)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run `python src/train.py` first.")

    print("Loading model...")
    model = keras.models.load_model(MODEL_PATH)

    print("Loading test data...")
    X_test, y_test = load_images_from_dir(TEST_DIR)
    print(f"Test set: {X_test.shape}")

    print("Running predictions...")
    y_scores = model.predict(X_test, verbose=1)
    y_pred = y_scores.argmax(axis=1).tolist()
    y_true = y_test.tolist()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=DISPLAY))

    os.makedirs(ASSETS_DIR, exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, os.path.join(ASSETS_DIR, "confusion_matrix.png"))
    plot_roc_curves(y_true, y_scores, os.path.join(ASSETS_DIR, "roc_curve.png"))

    macro_auc = roc_auc_score(
        label_binarize(y_true, classes=list(range(len(CLASSES)))),
        y_scores,
        average="macro",
        multi_class="ovr",
    )
    print(f"\nMacro ROC-AUC: {macro_auc:.4f}")
    print(f"\nPlots saved to {ASSETS_DIR}/")


if __name__ == "__main__":
    main()
