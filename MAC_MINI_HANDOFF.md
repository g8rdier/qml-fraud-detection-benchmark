# Mac Mini M4 Handoff — QML Noise Sweep

## Why this file exists

The depolarizing noise sweep (`run_noise.py`) was killed mid-run on the Fedora laptop
due to a charging failure (~16h into the run). Continuing on M4 Mac Mini.

---

## Project context (for Claude on Mac Mini)

This is a benchmark comparing quantum ML models (VQC, QSVM via PennyLane) against
classical baselines (XGBoost, Random Forest) on the Kaggle Credit Card Fraud dataset.

**Architecture:**
- `src/preprocessing.py` — RobustScaler → SMOTE → PCA (to n_qubits)
- `src/quantum_models.py` — VQCClassifier (StronglyEntanglingLayers) + QSVMClassifier (quantum kernel)
- `src/classical_models.py` — build_random_forest(), build_xgboost()
- `src/evaluation.py` — ModelMetrics, evaluate_model(), compare_models()
- `run_benchmark.py` — main benchmark entry point
- `run_noise.py` — depolarizing noise sweep (the task at hand)
- `run_latency.py` — per-sample inference timing
- `tests/` — 33 tests, all passing

**Key decisions:**
- RobustScaler (not StandardScaler) — financial data has extreme outliers
- SMOTE applied after train/test split on training set only (leakage prevention)
- PCA fitted on train only; n_components == n_qubits
- `lightning.qubit` backend for speed; `default.mixed` auto-selected when noise_level > 0
- Primary metrics: F1-fraud, PR-AUC, ROC-AUC, MCC

**Known bugs (already fixed in codebase):**
- Normalizer leakage: `_normalise_to_pi()` was re-fitting on predict — fixed, uses saved `self.normaliser_`
- QSVM class imbalance: `class_weight="balanced"` added to SVC
- VQC weight init: `normal(0, 0.1)` instead of `uniform(0, 2π)` to avoid barren plateaus

**Previous results (smoke test, 4 qubits, 30 epochs):**
- RF: F1-fraud=0.758, MCC=0.762
- XGBoost: F1-fraud=0.852, MCC=0.854
- QSVM (after fix): F1-fraud=0.884, MCC=0.867
- VQC: still needs more epochs to escape barren plateau

**Noise sweep progress before kill:**
- p=0.0   → VQC F1-fraud=0.699, QSVM F1-fraud=0.905 ✓
- p=0.001 → VQC F1-fraud=0.655, MCC=0.569, took ~4.9h on default.mixed ✓
- p=0.001 → QSVM kernel matrix was mid-compute when killed (no JSON saved)

Start the sweep from scratch — JSON only writes at end of full run.

---

## Setup on Mac Mini

**Python version: must be 3.12** (pyproject.toml requires `>=3.12,<3.13`)

```bash
# Install Python 3.12 if needed (via brew)
brew install python@3.12

# Clone repo
git clone https://github.com/g8rdier/qml-fraud-detection-benchmark.git
cd qml-fraud-detection-benchmark

# Create venv with Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies manually (pyproject.toml deps are pixi-only, not pip)
pip install numpy pandas scipy scikit-learn xgboost imbalanced-learn \
            pennylane pennylane-lightning matplotlib seaborn \
            joblib tqdm pyyaml jupyter ipykernel pytest pytest-cov

# Install package in editable mode
pip install -e . --no-deps
```

**Dataset:**
Download `creditcard.csv` from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place at: `data/raw/creditcard.csv` (284807 rows, 0.1727% fraud)

**Verify setup:**
```bash
pytest tests/ -v --ignore=tests/test_noise.py -x
```

---

## What to run

### Step 1 — Classical baselines (required by noise sweep)

```bash
python run_benchmark.py \
  --n-qubits 8 \
  --classical-only \
  --cv-folds 0 \
  --no-plots
```

Saves `results/models/random_forest.joblib` and `results/models/xgboost.joblib`.

### Step 2 — Noise sweep

```bash
mkdir -p results/noise
python run_noise.py \
  --data-path data/raw/creditcard.csv \
  --n-qubits 8 \
  > results/noise_run.log 2>&1 &

tail -f results/noise_run.log
```

Default noise levels: `[0.0, 0.001, 0.005, 0.01, 0.02, 0.05]`

Add `--vqc-only` to skip QSVM if you want faster results first.

**Expected outputs:**
- `results/noise/noise_results.json`
- `results/noise/noise_vs_metric.png`

---

## Key settings (inside run_noise.py)

| Parameter | Value |
|---|---|
| n_qubits | 8 |
| VQC epochs | 30 |
| VQC train samples | 200 |
| QSVM train samples | 100 |
| Max test samples | 500 |
| Backend (ideal) | lightning.qubit |
| Backend (noisy) | default.mixed (auto-selected) |

---

## Git workflow (important)

Follow https://github.com/g8rdier/dev-best-practices strictly:
- Feature branches: `type/short-description`
- Conventional commits: `feat:`, `fix:`, `chore:`, etc.
- Squash merge only: `gh pr merge <N> --squash --delete-branch --subject "feat: Title (#N)"`
- **No Co-Authored-By lines** in commits
- Commit author: `gregor.kobilarov@gmail.com`

**Configure git author on Mac Mini before committing:**
```bash
git config user.email "gregor.kobilarov@gmail.com"
git config user.name "g8rdier"
```

**Results go to the existing `feat/noise-model` branch:**
```bash
git checkout feat/noise-model
# copy/move results files then:
git add results/noise/noise_vs_metric.png
git commit -m "feat: Add noise sweep results and figure"
git push
```

**Next merge number is #7** (main currently has 6 squash merges).

---

## After the sweep — what's left for the full project

1. Commit `results/noise/noise_vs_metric.png` to `feat/noise-model` branch
   (note: `noise_results.json` is gitignored — only the PNG is committed)
2. Push → open PR → squash merge as **#7**: `feat: Add depolarizing noise sweep (#7)`
3. Final piece: `feat/report-notebook` — Jupyter notebook telling the full story
4. dev-best-practices repo (github.com/g8rdier/dev-best-practices) has issues #21–#26 open
