"""
Brain Tumor Classification — Streamlit App

Run with:
    streamlit run app/app.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import keras
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

from config import CLASSES, DISPLAY_NAMES, MODEL_PATH
from src.utils import preprocess_image

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main { max-width: 720px; margin: auto; }
    .result-box {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
    }
    .warning-box {
        background: #fefce8;
        border: 1px solid #fde047;
        border-radius: 10px;
        padding: 0.75rem 1.25rem;
        font-size: 0.85rem;
        margin-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Brain Tumor Classification")
st.caption(
    "MobileNet transfer learning · 98.24% validation accuracy · "
    "Classifies: Glioma, Meningioma, Pituitary, No Tumor"
)
st.divider()

# ── Model Loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return keras.models.load_model(MODEL_PATH)


model = load_model()

if model is None:
    st.error(
        f"Model file not found at `{MODEL_PATH}`.  "
        "Run `python src/train.py` to train and save the model first."
    )
    st.stop()

# ── Grad-CAM Helpers ───────────────────────────────────────────────────────────
def _find_last_conv_layer(base_model) -> str | None:
    """Return the name of the last layer with a 4D output in the base model."""
    for layer in reversed(base_model.layers):
        if len(layer.output.shape) == 4:
            return layer.name
    return None


def compute_gradcam(model, img_array: np.ndarray) -> np.ndarray | None:
    """
    Compute a Grad-CAM heatmap for the top predicted class.

    Returns a 2D float array in [0, 1], or None if the layer cannot be found.
    """
    base = model.layers[0]          # MobileNet functional sub-model
    conv_layer_name = _find_last_conv_layer(base)
    if conv_layer_name is None:
        return None

    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[base.get_layer(conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        pred_idx = tf.argmax(predictions[0])
        class_score = predictions[:, pred_idx]

    grads = tape.gradient(class_score, conv_outputs)          # (1, H, W, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))      # (C,)
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis] # (H, W, 1)
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-8
    return heatmap


def overlay_gradcam(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Resize heatmap to match image and blend with a JET colormap overlay."""
    h, w = image_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    blended = (image_rgb * (1 - alpha) + heatmap_rgb * alpha).clip(0, 255).astype(np.uint8)
    return blended

# ── Upload ─────────────────────────────────────────────────────────────────────
st.subheader("Upload MRI Scan")
st.write("Supported formats: JPG, PNG. For best results, use T1-weighted MRI scans.")

uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Could not decode the uploaded image. Please try a different file.")
        st.stop()

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ── Inference ──────────────────────────────────────────────────────────────
    with st.spinner("Analyzing scan..."):
        tensor = preprocess_image(image_bgr)
        scores = model.predict(tensor, verbose=0)[0]
        heatmap = compute_gradcam(model, tensor)

    predicted_idx = int(scores.argmax())
    predicted_class = CLASSES[predicted_idx]
    confidence = float(scores[predicted_idx]) * 100

    # ── Results row ────────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image_rgb, caption="Uploaded MRI", use_container_width=True)

    with col2:
        st.markdown("**Prediction**")
        st.markdown(
            f"<div class='result-box'>"
            f"<h3 style='margin:0'>{DISPLAY_NAMES[predicted_class]}</h3>"
            f"<p style='margin:4px 0 0; color:#16a34a'>{confidence:.1f}% confidence</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Per-class confidence**")

        display_labels = [DISPLAY_NAMES[c] for c in CLASSES]
        score_values = [float(scores[i]) * 100 for i in range(len(CLASSES))]
        colors = ["#16a34a" if i == predicted_idx else "#94a3b8" for i in range(len(CLASSES))]

        fig = go.Figure(
            go.Bar(
                x=score_values,
                y=display_labels,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.1f}%" for v in score_values],
                textposition="outside",
            )
        )
        fig.update_layout(
            xaxis=dict(range=[0, 110], showticklabels=False, showgrid=False),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=0, r=30, t=0, b=0),
            height=180,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Grad-CAM ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Model Explainability — Grad-CAM")
    st.caption(
        "Grad-CAM highlights which regions of the MRI the model focused on "
        "when making its prediction. Warmer colours = higher influence."
    )

    if heatmap is not None:
        overlay = overlay_gradcam(image_rgb, heatmap)
        cam_col1, cam_col2 = st.columns(2)
        with cam_col1:
            st.image(image_rgb, caption="Original", use_container_width=True)
        with cam_col2:
            st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
    else:
        st.info("Grad-CAM is not available for this model architecture.")

    # ── Class Description ──────────────────────────────────────────────────────
    with st.expander("What does this result mean?"):
        descriptions = {
            "glioma": (
                "**Glioma** tumors originate in the glial cells of the brain or spine. "
                "They are the most common primary brain tumor and range from slow-growing "
                "(grade I) to highly aggressive (grade IV glioblastoma)."
            ),
            "meningioma": (
                "**Meningioma** tumors arise from the meninges — the membranes surrounding "
                "the brain and spinal cord. Most are benign and slow-growing, though some "
                "can recur after treatment."
            ),
            "notumor": (
                "**No Tumor** detected. The model found no evidence of glioma, meningioma, "
                "or pituitary tumor in this scan. Always confirm with a licensed radiologist."
            ),
            "pituitary": (
                "**Pituitary** tumors form in the pituitary gland at the base of the brain. "
                "Most are benign adenomas, but they can affect hormone production and "
                "compress surrounding structures."
            ),
        }
        st.markdown(descriptions[predicted_class])

# ── Disclaimer ─────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='warning-box'>"
    "<strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only. "
    "It is not a substitute for professional medical diagnosis. "
    "The model recognizes only 4 of 150+ known brain tumor types and may produce incorrect results. "
    "Always consult a qualified medical professional."
    "</div>",
    unsafe_allow_html=True,
)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<br><p style='text-align:center; font-size:0.8rem; color:#94a3b8'>"
    "Dataset: Brain Tumor MRI Dataset · Msoud Nickparvar (Kaggle, 2021) · "
    "Model: MobileNet Transfer Learning"
    "</p>",
    unsafe_allow_html=True,
)
