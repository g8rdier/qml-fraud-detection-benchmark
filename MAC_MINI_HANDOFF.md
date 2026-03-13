# Mac Mini M4 Handoff — QML Noise Sweep

## Context

The depolarizing noise sweep (`run_noise.py`) was killed mid-run on the Fedora laptop
(charging failure, ~16h into the run). Pick it up fresh on the M4 Mac Mini.

**Progress made before kill:**
- p=0.0  → VQC F1-fraud=0.699, QSVM F1-fraud=0.905 ✓
- p=0.001 → VQC F1-fraud=0.655, MCC=0.569 ✓
- p=0.001 → QSVM kernel matrix was computing when killed (no result saved)

The JSON output (`results/noise/noise_results.json`) was **not written** — it only
writes at the end of the full sweep. Start from scratch.

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/qml-fraud-detection-benchmark.git
cd qml-fraud-detection-benchmark

# 2. Create venv (Python 3.13 preferred, 3.11+ works)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -e ".[dev]"
# or if pyproject.toml doesn't cover everything:
pip install pennylane pennylane-lightning scikit-learn xgboost imbalanced-learn \
            numpy pandas matplotlib joblib

# 4. Get the dataset
# Download creditcard.csv from Kaggle:
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place at: data/raw/creditcard.csv
```

---

## What to run

### Step 1 — Classical baselines (needed by noise sweep)

```bash
python run_benchmark.py \
  --n-qubits 8 \
  --classical-only \
  --cv-folds 0 \
  --no-plots
```

This saves `results/models/random_forest.joblib` and `results/models/xgboost.joblib`.

### Step 2 — Noise sweep

```bash
python run_noise.py \
  --data-path data/raw/creditcard.csv \
  --n-qubits 8 \
  > results/noise_run.log 2>&1 &

tail -f results/noise_run.log
```

Default noise levels: `[0.0, 0.001, 0.005, 0.01, 0.02, 0.05]`

**Expected runtime on M4:** VQC at p=0 (ideal, lightning.qubit) ~fast.
Noisy levels (default.mixed, density matrix) will be slower but M4 handles it well.

### Step 3 — After sweep completes

Outputs:
- `results/noise/noise_results.json`
- `results/noise/noise_vs_metric.png`

Copy the PNG back (or commit it) — it goes into `feat/noise-model` branch on the
Fedora machine (or wherever the main dev continues).

---

## Key settings

| Parameter | Value |
|---|---|
| n_qubits | 8 |
| VQC epochs | 30 (noise sweep uses reduced settings) |
| VQC train samples | 200 |
| QSVM train samples | 100 |
| Max test samples | 500 |
| Backend (ideal) | lightning.qubit |
| Backend (noisy) | default.mixed (auto-selected) |

---

## After the noise sweep — what's left

1. Commit `noise_results.json` + `noise_vs_metric.png` → push to `feat/noise-model` → PR → squash merge
2. `feat/report-notebook` — Jupyter notebook telling the full story (final deliverable)
3. dev-best-practices issues #21–#26 still open
