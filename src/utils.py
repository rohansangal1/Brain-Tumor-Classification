"""Shared preprocessing utilities."""

import os
import numpy as np
import cv2
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import IMAGE_SIZE, CLASS_LABELS


def load_images_from_dir(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk a directory structured as data_dir/<class_name>/<image>.jpg
    and return (X, y) arrays.
    """
    X, y = [], []
    for class_name, label in CLASS_LABELS.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in tqdm(os.listdir(class_dir), desc=f"Loading {class_name}"):
            fpath = os.path.join(class_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            X.append(img)
            y.append(label)
    return np.array(X), np.array(y)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Resize and reshape a single BGR image (as returned by cv2.imread)
    into the (1, IMAGE_SIZE, IMAGE_SIZE, 3) float array expected by the model.
    """
    resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LANCZOS4)
    return resized.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)
