# Hybrid 3‑Tier LSTM for Breast Cancer Classification (GitHub‑Ready)

This repository contains a **reproducible, modular pipeline** for tumor classification on the Breast Cancer dataset using a **Hybrid 3‑Tier LSTM** model. The pipeline includes:

- Clean data loading and preprocessing (drop `id` and `Unnamed: 32` if present)
- Encode target (`diagnosis`: Malignant **M** = 1, Benign **B** = 0)
- Train/test split with optional stratification
- **SMOTE on training set only**
- **Decision Tree** feature importance (with figure)
- **3‑Tier LSTM** (Dense → LSTM → Dense head) with input reshaped to `(samples, timesteps=1, features)`
- Full evaluation (Accuracy, Precision, Recall, F1, AUC, Confusion Matrix, ROC curve)
- **Explainability**: SHAP and LIME (with actual feature names)
- Reproducible configs via `config.yaml`
- One‑command run via `python main.py`

> **Note**: You need Python 3.10+ and a working TensorFlow environment. GPU is recommended but not required.

---

## Quickstart

```bash
# (optional) create venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

# Run the full pipeline
python main.py
```
# Use the ready conda environment
conda env create -f environment.yml
conda activate hybrid_lstm
python main.py

All outputs (models, figures, reports) will be saved under `outputs/`.

---

## Project Structure

```
.
├── config.yaml
├── main.py
├── requirements.txt
├── .gitignore
├── data/
│   └── Cancer_Data.xls
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── plots.py
│   └── models/
│       └── lstm_model.py
└── outputs/
    ├── figures/
    ├── models/
    └── reports/
```

---

## Reproducibility Notes

- SMOTE is applied **only** on the training split to avoid leakage.
- Feature scaling is fitted on train and applied to both train/test.
- Feature names are preserved throughout for correct plotting and explanations.
- SHAP uses a KernelExplainer on a small background sample to keep runtime reasonable on CPUs.
- LIME uses `LimeTabularExplainer` with consistent feature names and classes.

---
## Data set source
-https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

## Citations

