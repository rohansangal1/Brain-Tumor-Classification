"""Central configuration for the Brain Tumor Classification project."""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Testing")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")   # native Keras format, no pickle dependency
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# ── Classes ────────────────────────────────────────────────────────────────────
# Order matches the training label mapping from the notebook:
# labels = {'glioma':0, 'notumor':1, 'meningioma':2, 'pituitary':3}
CLASSES = ["glioma", "notumor", "meningioma", "pituitary"]
CLASS_LABELS = {name: idx for idx, name in enumerate(CLASSES)}
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary",
}

# ── Image Processing ───────────────────────────────────────────────────────────
IMAGE_SIZE = 150  # pixels (width and height)
CHANNELS = 3

# ── Training ───────────────────────────────────────────────────────────────────
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TEST_SPLIT = 0.2
RANDOM_STATE = 42
DROPOUT_RATE = 0.2
DENSE_UNITS = 512
