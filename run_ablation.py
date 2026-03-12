"""
run_ablation.py
===============
Qubit-scaling ablation study for the QML Fraud Detection Benchmark.

Sweeps n_qubits over a configurable range, running the full benchmark
(classical + quantum) at each point.  Produces:
  - results/ablation/ablation_results.json  — all metrics per qubit count
  - results/ablation/pca_variance.png       — PCA variance retained vs qubits
  - results/ablation/metric_vs_qubits.png   — key metrics vs qubits per model

Usage
-----
# Full sweep (slow — each point runs VQC + QSVM)
python run_ablation.py

# Classical models only (fast sanity check)
python run_ablation.py --classical-only

# Custom sweep range
python run_ablation.py --qubit-sweep 4 6 8 10 12
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.classical_models import build_random_forest, build_xgboost, train_classical_model
from src.evaluation import ModelMetrics, evaluate_model, find_optimal_threshold
from src.preprocessing import PreprocessingConfig, preprocess

ABLATION_DIR = Path("results/ablation")
DATA_PATH    = Path("data/raw/creditcard.csv")

DEFAULT_QUBIT_SWEEP = [4, 6, 8, 10, 12]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qubit-scaling ablation study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--qubit-sweep", type=int, nargs="+", default=DEFAULT_QUBIT_SWEEP)
    p.add_argument("--data-path", type=Path, default=DATA_PATH)
    p.add_argument("--classical-only", action="store_true")
    p.add_argument("--vqc-epochs", type=int, default=100)
    p.add_argument("--vqc-layers", type=int, default=2)
    p.add_argument("--quantum-backend", type=str, default="lightning.qubit")
    p.add_argument("--cv-folds", type=int, default=0,
                   help="CV folds for classical models (0 = skip; ablation is already slow)")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Single-point benchmark (one qubit count)
# ---------------------------------------------------------------------------

def _run_one(n_qubits: int, args: argparse.Namespace) -> dict:
    """
    Run the full benchmark for a single qubit count.

    Returns a dict:
        {
          "n_qubits": int,
          "pca_variance": float,           # fraction retained (0-1)
          "metrics": [ModelMetrics, ...]   # one per model
        }
    """
    logging.info("\n%s\n  n_qubits = %d\n%s", "=" * 50, n_qubits, "=" * 50)

    # ── Preprocessing ───────────────────────────────────────────────────────
    cfg = PreprocessingConfig(
        data_path=args.data_path,
        n_qubits=n_qubits,
        imbalance_strategy="smote",
    )
    data = preprocess(cfg)
    pca_variance = float(data.pca.explained_variance_ratio_.sum()) if data.pca else 1.0
    logging.info("PCA variance retained: %.2f%%", pca_variance * 100)

    results: list[ModelMetrics] = []

    # ── Classical models ────────────────────────────────────────────────────
    counts = data.class_counts_original
    spw    = counts[0] / counts[1]   # for XGBoost without SMOTE — not used here (smote)

    for name, clf in [
        ("Random Forest", build_random_forest()),
        ("XGBoost",       build_xgboost()),
    ]:
        tune = (name == "XGBoost")
        result = train_classical_model(
            model=clf,
            X_train=data.X_train, y_train=data.y_train,
            X_test=data.X_test,   y_test=data.y_test,
            X_val=data.X_val,     y_val=data.y_val,
            cv_folds=args.cv_folds,
            model_name=name.lower().replace(" ", "_"),
            tune_threshold=tune,
        )
        label = f"{name} (τ={result['threshold']:.3f})" if tune else name
        metrics = evaluate_model(
            model_name=label,
            y_true=data.y_test,
            y_pred=result["y_pred"],
            y_prob=result["y_prob"],
        )
        results.append(metrics)

    # ── Quantum models ───────────────────────────────────────────────────────
    if not args.classical_only:
        from src.quantum_models import QSVMClassifier, VQCClassifier

        MAX_VQC_TRAIN  = 600
        MAX_QSVM_TRAIN = 300
        MAX_QSVM_TEST  = 1000
        rng = np.random.default_rng(42)

        for name, clf in [
            ("VQC",  VQCClassifier(n_qubits=n_qubits, n_layers=args.vqc_layers,
                                   n_epochs=args.vqc_epochs, backend=args.quantum_backend)),
            ("QSVM", QSVMClassifier(n_qubits=n_qubits, backend=args.quantum_backend)),
        ]:
            X_tr, y_tr = data.X_train, data.y_train
            X_te, y_te = data.X_test,  data.y_test

            cap = MAX_VQC_TRAIN if name == "VQC" else MAX_QSVM_TRAIN
            if len(X_tr) > cap:
                idx0, idx1 = np.where(y_tr == 0)[0], np.where(y_tr == 1)[0]
                half = cap // 2
                idx = np.concatenate([
                    rng.choice(idx0, size=min(half, len(idx0)), replace=False),
                    rng.choice(idx1, size=min(half, len(idx1)), replace=False),
                ])
                X_tr, y_tr = X_tr[idx], y_tr[idx]

            if name == "QSVM" and len(X_te) > MAX_QSVM_TEST:
                idx0, idx1 = np.where(y_te == 0)[0], np.where(y_te == 1)[0]
                half = MAX_QSVM_TEST // 2
                idx = np.concatenate([
                    rng.choice(idx0, size=min(half, len(idx0)), replace=False),
                    rng.choice(idx1, size=min(half, len(idx1)), replace=False),
                ])
                X_te, y_te = X_te[idx], y_te[idx]

            logging.info("=== Training %s (n_qubits=%d) ===", name, n_qubits)
            clf.fit(X_tr, y_tr)

            if name == "VQC":
                logging.info("VQC threshold tuning on full val set (%d samples)…", len(data.y_val))
                val_prob  = clf.predict_proba(data.X_val)[:, 1]
                threshold = find_optimal_threshold(data.y_val, val_prob)
                logging.info("VQC tuned threshold: %.4f", threshold)
                y_prob = clf.predict_proba(X_te)
                y_pred = (y_prob[:, 1] >= threshold).astype(int)
                label  = f"VQC (τ={threshold:.3f})"
            else:
                y_prob = clf.predict_proba(X_te)
                y_pred = clf.predict(X_te)
                label  = "QSVM"

            results.append(evaluate_model(
                model_name=label, y_true=y_te, y_pred=y_pred, y_prob=y_prob,
            ))

    return {"n_qubits": n_qubits, "pca_variance": pca_variance, "metrics": results}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_pca_variance(sweep: list[int], variances: list[float], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep, [v * 100 for v in variances], marker="o", linewidth=2)
    ax.axhline(80, color="red", linestyle="--", linewidth=1, label="80% threshold")
    ax.set_xlabel("Number of qubits (PCA components)")
    ax.set_ylabel("Variance retained (%)")
    ax.set_title("PCA Variance Retained vs Qubit Count (post-SMOTE)")
    ax.set_xticks(sweep)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved %s", save_path)


def _plot_metric_vs_qubits(
    sweep: list[int],
    all_results: list[dict],
    save_path: Path,
) -> None:
    # Collect model names from first sweep point (consistent across points)
    model_names = [m.model_name for m in all_results[0]["metrics"]]
    # Strip tuned-threshold suffix for a stable legend key
    def _base(name: str) -> str:
        return name.split(" (")[0]

    base_names = list(dict.fromkeys(_base(n) for n in model_names))
    metrics_to_plot = ["f1_fraud", "pr_auc", "roc_auc", "mcc"]
    titles = {"f1_fraud": "F1-Fraud", "pr_auc": "PR-AUC",
              "roc_auc": "ROC-AUC", "mcc": "MCC"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics_to_plot):
        for base in base_names:
            values = []
            for point in all_results:
                # Find the metric for this model at this qubit count
                match = next(
                    (m for m in point["metrics"] if _base(m.model_name) == base), None
                )
                values.append(getattr(match, metric) if match else float("nan"))
            ax.plot(sweep, values, marker="o", linewidth=2, label=base)

        ax.set_title(titles[metric])
        ax.set_ylabel("Score")
        ax.set_xticks(sweep)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel("Number of qubits")

    fig.suptitle(
        "Model Performance vs Qubit Count — QML Fraud Detection Benchmark",
        fontsize=13,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved %s", save_path)


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

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  QML Fraud Detection — Qubit Ablation Study          ║")
    print(f"║  Sweep: {args.qubit_sweep}                        ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    try:
        preprocess(PreprocessingConfig(data_path=args.data_path, n_qubits=4))
    except FileNotFoundError as exc:
        logging.error(str(exc))
        sys.exit(1)

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    t_total = time.perf_counter()

    for n_qubits in args.qubit_sweep:
        t0 = time.perf_counter()
        point = _run_one(n_qubits, args)
        elapsed = time.perf_counter() - t0
        logging.info("n_qubits=%d done in %.1f min", n_qubits, elapsed / 60)
        all_results.append(point)

    logging.info("Total ablation time: %.1f min", (time.perf_counter() - t_total) / 60)

    # ── Save JSON ────────────────────────────────────────────────────────────
    out_json = ABLATION_DIR / "ablation_results.json"
    serialisable = [
        {
            "n_qubits":     p["n_qubits"],
            "pca_variance": p["pca_variance"],
            "metrics":      [m.to_dict() for m in p["metrics"]],
        }
        for p in all_results
    ]
    with open(out_json, "w") as fh:
        json.dump(serialisable, fh, indent=2)
    logging.info("Ablation results saved to %s", out_json)

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n\nAblation Summary — F1-Fraud per model\n")
    model_bases = list(dict.fromkeys(
        m.model_name.split(" (")[0] for m in all_results[0]["metrics"]
    ))
    header = f"{'Qubits':>7}  {'Var%':>6}  " + "  ".join(f"{b:>14}" for b in model_bases)
    print(header)
    print("-" * len(header))
    for p in all_results:
        row = f"{p['n_qubits']:>7}  {p['pca_variance']*100:>5.1f}%  "
        for base in model_bases:
            m = next((x for x in p["metrics"] if x.model_name.split(" (")[0] == base), None)
            row += f"  {m.f1_fraud:>14.4f}" if m else f"  {'N/A':>14}"
        print(row)
    print()

    # ── Plots ────────────────────────────────────────────────────────────────
    if not args.no_plots:
        sweep     = [p["n_qubits"]    for p in all_results]
        variances = [p["pca_variance"] for p in all_results]
        _plot_pca_variance(sweep, variances, ABLATION_DIR / "pca_variance.png")
        _plot_metric_vs_qubits(sweep, all_results, ABLATION_DIR / "metric_vs_qubits.png")

    print(f"Ablation complete. Results saved to {ABLATION_DIR}/")


if __name__ == "__main__":
    main()
