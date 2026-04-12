# Brain Tumor Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-98.24%25-brightgreen)]()

An end-to-end deep learning pipeline for MRI-based brain tumor classification, achieving **98.24% validation accuracy** via MobileNet transfer learning — deployed as an interactive Streamlit web application.

---

## Overview

Brain tumors affect over **200,000 Americans annually** and require rapid, accurate diagnosis to inform treatment. Manual MRI interpretation is time-intensive and prone to inter-rater variability. This project demonstrates how a fine-tuned convolutional neural network can serve as a reliable first-pass screening tool, classifying MRI scans into four clinically relevant categories in under a second.

> **Medical Disclaimer:** This tool is intended for research and educational purposes only. It is not a substitute for professional medical diagnosis.

---

## Demo

> Upload an MRI scan and receive an instant tumor classification with per-class confidence scores.

![App Demo Placeholder](assets/demo_placeholder.png)

*To run the app locally, see [Usage](#usage).*

---

## Features

- Classifies MRI scans into 4 categories: **Glioma**, **Meningioma**, **Pituitary**, **No Tumor**
- 98.24% validation accuracy on 5,608 training images
- Benchmarked against 7 models: Logistic Regression, Ridge, Random Forest, Decision Tree, SVM, KNN, CNN
- Interactive Streamlit app with real-time confidence visualization
- Clean, modular codebase — training, inference, and app logic fully separated

---

## Model Architecture

| Component         | Detail                                      |
|-------------------|---------------------------------------------|
| Base Model        | MobileNet (ImageNet pretrained)             |
| Input Size        | 150 × 150 × 3                               |
| Head              | GlobalMaxPool → Dropout(0.2) → Dense(512, ReLU) → Dropout(0.2) → Dense(4, Softmax) |
| Optimizer         | Adam                                        |
| Loss              | Sparse Categorical Crossentropy             |
| Epochs            | 30                                          |
| Training Set      | 5,608 images                                |

MobileNet was selected for its parameter efficiency: it achieves near-ResNet performance at a fraction of the computational cost, making it practical for deployment on consumer hardware.

---

## Dataset

**Brain Tumor MRI Dataset** — Msoud Nickparvar (Kaggle, 2021)

| Split    | Images |
|----------|--------|
| Training | 5,608  |
| Testing  | 1,311  |
| **Total**| **6,919** |

| Class       | Train | Test |
|-------------|-------|------|
| Glioma      | 1,321 | 300  |
| Meningioma  | 1,339 | 306  |
| No Tumor    | 1,595 | 405  |
| Pituitary   | 1,457 | 300  |

---

## Results

### Model Comparison

| Model               | Accuracy  |
|---------------------|-----------|
| Logistic Regression | ~84%      |
| Ridge Classifier    | ~78%      |
| Random Forest       | ~82%      |
| Decision Tree       | ~73%      |
| SVM                 | ~86%      |
| KNN                 | ~83%      |
| Custom CNN          | ~89%      |
| AlexNet             | ~93%      |
| **MobileNet**       | **98.24%**|

### MobileNet Performance

| Metric           | Score  |
|------------------|--------|
| Accuracy         | 98.24% |
| Precision        | ~0.983 |
| Recall           | ~0.982 |
| F1 Score         | ~0.982 |
| Cohen Kappa      | ~0.976 |
| ROC-AUC (macro)  | ~0.999 |

---

## Tech Stack

- **Deep Learning:** TensorFlow / Keras, MobileNet
- **Classical ML:** scikit-learn
- **Data Processing:** NumPy, OpenCV, Pillow
- **Visualization:** Matplotlib, Seaborn, Plotly
- **App:** Streamlit
- **Notebook:** Google Colab

---

## Project Structure

```
brain-tumor-classification/
├── app/
│   └── app.py                  # Streamlit web application
├── src/
│   ├── train.py                # Model training script
│   ├── predict.py              # Inference script (CLI)
│   └── utils.py                # Shared preprocessing utilities
├── models/
│   └── model.joblib            # Saved MobileNet model (not tracked in git)
├── data/
│   ├── Training/               # Training images by class
│   └── Testing/                # Test images by class
├── notebooks/
│   └── brain_tumor_classification.ipynb
├── assets/                     # Screenshots, demo GIFs
├── config.py                   # Central configuration
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/rohansangal1/Brain-Tumor-Classification.git
cd Brain-Tumor-Classification
pip install -r requirements.txt
```

Requires Python 3.10+.

---

## Usage

### Run the Streamlit App

```bash
streamlit run app/app.py
```

Navigate to `http://localhost:8501`, upload an MRI image (`.jpg` or `.png`), and view the prediction with confidence scores.

### Train the Model

```bash
python src/train.py
```

By default reads data from `data/Training/` and saves the model to `models/model.joblib`. Edit `config.py` to change paths or hyperparameters.

### Run Inference on a Single Image

```bash
python src/predict.py --image path/to/mri.jpg
```

---

## Limitations

- **4 classes only.** There are over 150 known brain tumor types. Any tumor not matching these four classes will be force-classified into the nearest category.
- **MRI modality.** The model was trained exclusively on T1-weighted MRI scans. CT scans or other modalities may produce unreliable results.
- **Dataset size.** ~7,000 images is small by medical imaging standards. Production-grade systems require orders of magnitude more data.
- **No uncertainty quantification.** The model outputs softmax probabilities but is not calibrated — high confidence does not guarantee correctness.
- **Not FDA-cleared.** This is a research prototype, not a medical device.

---

## Future Work

- [ ] Expand to additional tumor classes using broader datasets (e.g., BraTS)
- [ ] Add Grad-CAM heatmap overlays to explain which regions influenced the prediction
- [ ] Calibrate output probabilities for clinical reliability
- [ ] Experiment with EfficientNet and Vision Transformer architectures
- [ ] Containerize with Docker for portable deployment

---

## Citation

```
Msoud Nickparvar. (2021). Brain Tumor MRI Dataset [Data set].
Kaggle. https://doi.org/10.34740/KAGGLE/DSV/2645886
```

---

## License

This project is licensed under the [MIT License](LICENSE).
