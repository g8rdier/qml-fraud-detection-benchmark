# QML Fraud Detection Benchmark

> **Client:** Leading Financial Service Provider
> **Objective:** Evaluate the practical utility of Hybrid Quantum-Classical Machine Learning for real-time financial fraud detection in the NISQ era.

---

## Overview

This repository implements a rigorous benchmarking framework that compares:

| Model | Type | Library |
|---|---|---|
| Random Forest | Classical | scikit-learn |
| XGBoost | Classical | xgboost |
| Variational Quantum Classifier (VQC) | Quantum | PennyLane |
| Quantum Support Vector Machine (QSVM) | Quantum | PennyLane + scikit-learn |

The benchmark is designed around the challenges specific to financial fraud detection:
- **Extreme class imbalance** (~0.17% fraud in the reference dataset)
- **High dimensionality** vs. the limited qubit count of NISQ simulators
- **Rigorous metrics** that reflect real-world fraud detection performance

---

## Project Structure

```
qml-fraud-detection-benchmark/
├── data/
│   ├── raw/                  # Place creditcard.csv here
│   └── processed/            # Auto-generated preprocessed arrays
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── results/
│   ├── figures/              # Saved plots
│   └── metrics/              # JSON metric reports
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Dataset verification & download instructions
│   ├── preprocessing.py      # Scaling · SMOTE · PCA pipeline
│   ├── classical_models.py   # Random Forest & XGBoost builders
│   ├── quantum_models.py     # VQC & QSVM implementations (PennyLane)
│   └── evaluation.py         # Metrics, plots, comparison tables
├── tests/
│   ├── test_preprocessing.py
│   ├── test_quantum_models.py
│   └── test_evaluation.py
├── requirements.txt
└── README.md
```

---

## Dataset

The benchmark uses the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset.

**Download steps:**
```bash
# Option A – Kaggle CLI
kaggle datasets download -d mlg-ulb/creditcardfraud --path data/raw --unzip

# Option B – Manual
# Download creditcard.csv from Kaggle and place it at data/raw/creditcard.csv
```

---

## Setup

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Preprocessing Pipeline

The `src/preprocessing.py` module addresses the core data challenges:

| Challenge | Solution |
|---|---|
| Class imbalance (~0.17% fraud) | SMOTE oversampling (training set only) or class-weight |
| Outliers in transaction amounts | `RobustScaler` (median/IQR-based) |
| High dimensionality (30 features) vs. qubit limit | PCA to 4–8 components |
| Data leakage prevention | PCA & scaler fitted on train split only |

```python
from src.preprocessing import PreprocessingConfig, preprocess

cfg = PreprocessingConfig(
    data_path="data/raw/creditcard.csv",
    n_qubits=8,                      # → 8 PCA components
    imbalance_strategy="smote",
)
data = preprocess(cfg)
# data.X_train  shape: (n_train, 8)  — qubit-ready
# data.X_test   shape: (n_test,  8)
```

---

## Quantum Models

### Variational Quantum Classifier (VQC)

```
AngleEmbedding(X) → StronglyEntanglingLayers(weights) → ⟨Z₀⟩
```

- Input: PCA-reduced features normalised to `[0, π]`
- Ansatz: `StronglyEntanglingLayers` (expressibility optimised for NISQ)
- Optimiser: Adam (PennyLane autograd)

### Quantum SVM (QSVM)

- Computes a quantum kernel matrix K(x, x') = |⟨φ(x)|φ(x')⟩|²
- Feeds the kernel to a scikit-learn `SVC(kernel="precomputed")`
- Feature map: double `AngleEmbedding` (ZZ-style, captures 2nd-order interactions)

---

## Evaluation Metrics

Accuracy is reported as a reference only.  Primary metrics:

| Metric | Why it matters for fraud detection |
|---|---|
| **F1 (fraud class)** | Balances precision and recall for the minority class |
| **PR-AUC** | Summarises the precision-recall trade-off across all thresholds |
| **ROC-AUC** | Overall discriminative ability |
| **MCC** | Single balanced score accounting for all four confusion-matrix cells |

---

## Running Tests

```bash
pytest tests/ -v --cov=src
```

---

## NISQ Considerations

- All quantum circuits are simulated using `lightning.qubit` (high-performance CPU backend).
- The qubit budget is set to **4–8 qubits** to keep simulation tractable.
- PCA dimensionality reduction is the key bridge between the 30-feature financial dataset and the qubit constraint.
- Noise models and hardware transpilation are out of scope for this benchmark phase.

---

## Roadmap

- [ ] Run full benchmark and save results to `results/metrics/`
- [ ] Exploratory analysis notebook (`notebooks/01_exploratory_analysis.ipynb`)
- [ ] Hardware execution on IBM Quantum (via PennyLane `qiskit.ibmq` plugin)
- [ ] Autoencoder-based dimensionality reduction as PCA alternative
- [ ] Noise-aware VQC training with depolarising error model
