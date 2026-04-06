"""
run_benchmark.py
================
Main entry point for the QML Fraud Detection Benchmark.

Usage
-----
# Full benchmark (classical + quantum — quantum models are slow)
python run_benchmark.py

# Classical models only (fast, good for CI / iteration)
python run_benchmark.py --classical-only

# Custom qubit count and imbalance strategy
python run_benchmark.py --n-qubits 4 --imbalance smote --cv-folds 3

# Skip plots (headless / CI environments)
python run_benchmark.py --no-plots
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ── Project imports ────────────────────────────────────────────────────────
from src.classical_models import (
    build_random_forest,
    build_xgboost,
    train_classical_model,
)
from src.evaluation import (
    ModelMetrics,
    compare_models,
    evaluate_model,
    plot_calibration_curves,
    plot_confusion_matrices,
    plot_confusion_matrices_quantum_focus,
    plot_metric_comparison,
    plot_pr_curves,
    plot_roc_curves,
    save_metrics_json,
)
from src.preprocessing import PreprocessingConfig, preprocess

# ── Paths ──────────────────────────────────────────────────────────────────
RESULTS_DIR   = Path("results")
METRICS_DIR   = RESULTS_DIR / "metrics"
FIGURES_DIR   = RESULTS_DIR / "figures"
MODELS_DIR    = RESULTS_DIR / "models"
DATA_PATH     = Path("data/raw/creditcard.csv")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QML vs Classical Fraud Detection Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-path", type=Path, default=DATA_PATH,
        help="Path to the raw creditcard.csv dataset.",
    )
    p.add_argument(
        "--n-qubits", type=int, default=8,
        help="Number of PCA components / qubits for quantum models.",
    )
    p.add_argument(
        "--imbalance", choices=["smote", "class_weight", "none"],
        default="smote",
        help="Strategy for handling class imbalance.",
    )
    p.add_argument(
        "--cv-folds", type=int, default=5,
        help="Stratified CV folds for classical models. 0 = skip CV.",
    )
    p.add_argument(
        "--classical-only", action="store_true",
        help="Skip quantum models (much faster, useful for debugging).",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Suppress figure generation (useful in headless environments).",
    )
    p.add_argument(
        "--vqc-epochs", type=int, default=100,
        help="Training epochs for the VQC.",
    )
    p.add_argument(
        "--vqc-layers", type=int, default=2,
        help="Number of StronglyEntanglingLayers in the VQC.",
    )
    p.add_argument(
        "--quantum-backend", type=str, default="lightning.qubit",
        help="PennyLane device backend for quantum models.",
    )
    p.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO",
    )
    p.add_argument(
        "--save-predictions", action="store_true",
        help="Save raw per-sample predictions alongside metrics JSON (enables plot regeneration).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _run_classical(
    data,
    args: argparse.Namespace,
    results: list[ModelMetrics],
    plot_data: list[dict],
) -> None:
    """Fit Random Forest and XGBoost, collect metrics."""
    imbalance = args.imbalance

    # Derive scale_pos_weight for XGBoost when SMOTE is NOT used
    counts = data.class_counts_original
    spw = counts[0] / counts[1] if imbalance != "smote" else 1.0

    models = {
        "Random Forest": build_random_forest(
            class_weight="balanced" if imbalance == "class_weight" else None
        ),
        "XGBoost": build_xgboost(scale_pos_weight=spw),
    }

    for name, clf in models.items():
        logging.info("\n%s", "=" * 60)
        # Tune decision threshold for XGBoost: SMOTE balances training data
        # but the real test distribution is ~0.17% fraud, making the default
        # 0.5 threshold too conservative.  Tune on a held-out validation split.
        tune = (name == "XGBoost" and imbalance == "smote")
        result = train_classical_model(
            model=clf,
            X_train=data.X_train,
            y_train=data.y_train,
            X_test=data.X_test,
            y_test=data.y_test,
            X_val=data.X_val,
            y_val=data.y_val,
            cv_folds=args.cv_folds,
            save_dir=MODELS_DIR,
            model_name=name.lower().replace(" ", "_"),
            tune_threshold=tune,
        )

        label = f"{name} (tuned τ={result['threshold']:.3f})" if tune else name
        metrics = evaluate_model(
            model_name=label,
            y_true=data.y_test,
            y_pred=result["y_pred"],
            y_prob=result["y_prob"],
        )
        results.append(metrics)
        plot_data.append({
            "name":   label,
            "y_true": data.y_test,
            "y_pred": result["y_pred"],
            "y_prob": result["y_prob"][:, 1],
        })
        logging.info("Fit time: %.2f s | threshold: %.4f",
                     result["fit_time"], result["threshold"])


def _run_quantum(
    data,
    args: argparse.Namespace,
    results: list[ModelMetrics],
    plot_data: list[dict],
) -> None:
    """Fit VQC and QSVM, collect metrics."""
    # Lazy import — avoids penalising --classical-only runs with PennyLane overhead
    from src.evaluation import find_optimal_threshold
    from src.quantum_models import QSVMClassifier, VQCClassifier

    quantum_models = {
        "VQC": VQCClassifier(
            n_qubits=args.n_qubits,
            n_layers=args.vqc_layers,
            n_epochs=args.vqc_epochs,
            backend=args.quantum_backend,
        ),
        "QSVM": QSVMClassifier(
            n_qubits=args.n_qubits,
            backend=args.quantum_backend,
        ),
    }

    # Subsample limits — quantum simulation is expensive:
    #   VQC  cost() is O(n_train · epochs) circuit evals
    #   QSVM kernel is O(n_train²) train + O(n_test · n_train) predict
    # Both are intractable on the full dataset without these caps.
    MAX_VQC_TRAIN  = 600   # 300 fraud + 300 legit, stratified
    MAX_QSVM_TRAIN = 300   # 300×300 = 90k kernel evals (train)
    MAX_QSVM_TEST  = 1000  # 1000×300 = 300k kernel evals (predict)
    n_train = len(data.X_train)
    n_test  = len(data.X_test)

    for name, clf in quantum_models.items():
        logging.info("\n%s", "=" * 60)
        logging.info("=== Training %s ===", name)

        X_tr, y_tr = data.X_train, data.y_train
        X_te, y_te = data.X_test,  data.y_test
        rng = np.random.default_rng(42)

        # ── Stratified train subsample ──────────────────────────────────────
        cap = MAX_VQC_TRAIN if name == "VQC" else MAX_QSVM_TRAIN
        if n_train > cap:
            logging.warning(
                "%s: subsampling train %d → %d (stratified).", name, n_train, cap,
            )
            idx0 = np.where(y_tr == 0)[0]
            idx1 = np.where(y_tr == 1)[0]
            half = cap // 2
            idx = np.concatenate([
                rng.choice(idx0, size=min(half, len(idx0)), replace=False),
                rng.choice(idx1, size=min(half, len(idx1)), replace=False),
            ])
            X_tr, y_tr = X_tr[idx], y_tr[idx]

        # ── Stratified test subsample (QSVM only — O(n_test·n_train) kernel) ─
        if name == "QSVM" and n_test > MAX_QSVM_TEST:
            logging.warning(
                "QSVM: subsampling test %d → %d (stratified) for kernel predict.",
                n_test, MAX_QSVM_TEST,
            )
            idx0 = np.where(y_te == 0)[0]
            idx1 = np.where(y_te == 1)[0]
            half = MAX_QSVM_TEST // 2
            idx = np.concatenate([
                rng.choice(idx0, size=min(half, len(idx0)), replace=False),
                rng.choice(idx1, size=min(half, len(idx1)), replace=False),
            ])
            X_te, y_te = X_te[idx], y_te[idx]

        t0 = time.perf_counter()
        clf.fit(X_tr, y_tr)
        fit_time = time.perf_counter() - t0
        logging.info("Fit completed in %.2f s", fit_time)

        # ── Threshold tuning ────────────────────────────────────────────────
        # VQC: tune on the full pre-SMOTE val set (real 0.17% fraud rate).
        #   Subsampling val legit destroys calibration — threshold tuned on an
        #   artificially balanced set is far too aggressive at real fraud rates.
        #   Cost: ~7 min of predict_proba calls, acceptable after 12 min of training.
        #
        # QSVM: skip threshold tuning. class_weight="balanced" in the SVC already
        #   corrects for class imbalance, and tuning on a subsampled val set hurt
        #   performance (0.849 tuned vs 0.857 untuned in testing).
        if name == "VQC":
            logging.info(
                "VQC threshold tuning on full val set (%d samples, real distribution)…",
                len(data.y_val),
            )
            val_prob = clf.predict_proba(data.X_val)[:, 1]
            threshold = find_optimal_threshold(data.y_val, val_prob)
            logging.info("VQC tuned threshold: %.4f (was 0.5000)", threshold)
            y_prob = clf.predict_proba(X_te)
            y_pred = (y_prob[:, 1] >= threshold).astype(int)
            label  = f"{name} (tuned τ={threshold:.3f})"
        else:
            # QSVM — rely on class_weight="balanced" for calibration
            y_prob = clf.predict_proba(X_te)
            y_pred = clf.predict(X_te)
            label  = name

        metrics = evaluate_model(
            model_name=label,
            y_true=y_te,
            y_pred=y_pred,
            y_prob=y_prob,
        )
        results.append(metrics)
        plot_data.append({
            "name":   label,
            "y_true": y_te,
            "y_pred": y_pred,
            "y_prob": y_prob[:, 1],
        })


# ---------------------------------------------------------------------------
# Predictions persistence (enables offline plot regeneration)
# ---------------------------------------------------------------------------

def _save_predictions(plot_data: list[dict], path: Path) -> None:
    """Persist per-sample predictions so plots can be regenerated without re-running."""
    import json as _json
    serialisable = [
        {
            "name":   m["name"],
            "y_true": m["y_true"].tolist(),
            "y_pred": m["y_pred"].tolist(),
            "y_prob": m["y_prob"].tolist(),
        }
        for m in plot_data
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        _json.dump(serialisable, fh)
    logging.info("Predictions saved to %s", path)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _generate_plots(plot_data: list[dict], results: list[ModelMetrics]) -> None:
    logging.info("\nGenerating figures → %s", FIGURES_DIR)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_pr_curves(plot_data,        save_path=FIGURES_DIR / "pr_curves.png")
    plot_roc_curves(plot_data,       save_path=FIGURES_DIR / "roc_curves.png")
    plot_confusion_matrices(plot_data, save_path=FIGURES_DIR / "confusion_matrices.png")
    plot_confusion_matrices_quantum_focus(plot_data, save_path=FIGURES_DIR / "confusion_matrices_quantum_focus.png")
    plot_metric_comparison(results,  save_path=FIGURES_DIR / "metric_comparison.png")
    plot_calibration_curves(plot_data, save_path=FIGURES_DIR / "calibration_curves.png")

    logging.info("Figures saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    banner = (
        "\n"
        "╔══════════════════════════════════════════════════════╗\n"
        "║  QML Fraud Detection Benchmark                       ║\n"
        "║  Leading Financial Service Provider                  ║\n"
        "╚══════════════════════════════════════════════════════╝\n"
    )
    print(banner)

    # ── 1. Preprocessing ───────────────────────────────────────────────────
    logging.info("Step 1/3 — Preprocessing")
    cfg = PreprocessingConfig(
        data_path=args.data_path,
        n_qubits=args.n_qubits,
        imbalance_strategy=args.imbalance,
    )
    try:
        data = preprocess(cfg)
    except FileNotFoundError as exc:
        logging.error(str(exc))
        sys.exit(1)

    logging.info(
        "Dataset ready | train: %d samples | test: %d samples | features: %d",
        len(data.X_train), len(data.X_test), data.n_features_final,
    )

    # ── 2. Training ────────────────────────────────────────────────────────
    results:   list[ModelMetrics] = []
    plot_data: list[dict]         = []

    logging.info("\nStep 2/3 — Model Training")
    _run_classical(data, args, results, plot_data)

    if not args.classical_only:
        _run_quantum(data, args, results, plot_data)
    else:
        logging.info("--classical-only flag set: skipping quantum models.")

    # ── 3. Results ─────────────────────────────────────────────────────────
    logging.info("\nStep 3/3 — Results")

    print("\n")
    compare_models(results)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    metrics_path = METRICS_DIR / f"benchmark_{timestamp}.json"
    save_metrics_json(results, metrics_path)

    if args.save_predictions:
        _save_predictions(plot_data, METRICS_DIR / f"predictions_{timestamp}.json")

    if not args.no_plots:
        _generate_plots(plot_data, results)

    print(f"\nBenchmark complete. Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
