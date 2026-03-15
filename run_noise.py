"""
run_noise.py
============
Depolarizing noise sweep for the QML Fraud Detection Benchmark.

Trains VQC and QSVM at each noise level and compares their performance
against ideal (noise_level=0) and classical baselines.  Answers:

  "How much does performance degrade as hardware noise increases?"
  "At what noise level do quantum models drop below classical baselines?"

Noise model
-----------
DepolarizingChannel(p) inserted after AngleEmbedding and after the
variational/kernel layers.  At p=0 the circuit is ideal.  Typical NISQ
device error rates: single-qubit gates ~0.1% (p≈0.001), two-qubit ~1%
(p≈0.01).  Current best hardware: p≈0.001.

Output
------
results/noise/noise_results.json      — metrics per model per noise level
results/noise/noise_vs_metric.png     — F1-fraud & MCC vs noise level

Usage
-----
python run_noise.py                        # full sweep
python run_noise.py --noise-levels 0 0.01  # custom levels
python run_noise.py --vqc-only             # skip QSVM (faster)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation import evaluate_model, find_optimal_threshold
from src.preprocessing import PreprocessingConfig, preprocess

NOISE_DIR  = Path("results/noise")
MODELS_DIR = Path("results/models")
DATA_PATH  = Path("data/raw/creditcard.csv")
MERGE_DIR  = Path("results/noise")

DEFAULT_NOISE_LEVELS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]

# Reduced training sizes to keep the sweep tractable
VQC_TRAIN_SAMPLES  = 200
VQC_EPOCHS         = 30
QSVM_TRAIN_SAMPLES = 100
MAX_TEST_SAMPLES   = 500


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Depolarizing noise sweep for quantum models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ── merge subcommand ──────────────────────────────────────────────────────
    m = sub.add_parser("merge", help="Merge per-level JSONs into one and plot.")
    m.add_argument("--noise-dirs", type=Path, nargs="+", required=True,
                   help="Directories containing noise_results.json files to merge.")
    m.add_argument("--out-dir",    type=Path, default=MERGE_DIR)
    m.add_argument("--no-plots",   action="store_true")

    # ── sweep subcommand (default) ────────────────────────────────────────────
    p = sub.add_parser("sweep", help="Run noise sweep (default).")
    p.add_argument("--data-path",    type=Path,  default=DATA_PATH)
    p.add_argument("--n-qubits",     type=int,   default=8)
    p.add_argument(
        "--noise-levels", type=float, nargs="+", default=DEFAULT_NOISE_LEVELS,
        help="Depolarizing error probabilities to sweep.",
    )
    p.add_argument("--noise-dir",  type=Path, default=NOISE_DIR,
                   help="Output directory for this run's results.")
    p.add_argument("--vqc-only",   action="store_true", help="Skip QSVM (faster).")
    p.add_argument("--no-plots",   action="store_true")
    p.add_argument("--log-level",  choices=["DEBUG", "INFO", "WARNING"], default="INFO")

    args = parser.parse_args()
    # default to sweep when no subcommand given (backwards compatible)
    if args.command is None:
        args = p.parse_args()
        args.command = "sweep"
    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subsample_stratified(
    X: np.ndarray, y: np.ndarray, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    half = n // 2
    idx = np.concatenate([
        rng.choice(idx0, size=min(half, len(idx0)), replace=False),
        rng.choice(idx1, size=min(half, len(idx1)), replace=False),
    ])
    return X[idx], y[idx]


def _classical_baselines(models_dir: Path, X_test, y_test) -> list[dict]:
    """Evaluate saved classical models — noise-independent."""
    results = []
    for name, fname in [("Random Forest", "random_forest.joblib"),
                        ("XGBoost",       "xgboost.joblib")]:
        path = models_dir / fname
        if not path.exists():
            logging.warning("Saved model not found: %s — skipping.", path)
            continue
        clf = joblib.load(path)
        y_prob = clf.predict_proba(X_test)
        y_pred = (y_prob[:, 1] >= 0.5).astype(int)
        m = evaluate_model(name, y_test, y_pred, y_prob)
        results.append({"model": name, "noise_level": None, "metrics": m.to_dict()})
    return results


# ---------------------------------------------------------------------------
# Single noise-level run
# ---------------------------------------------------------------------------

def _run_vqc(X_train, y_train, X_val, y_val, X_test, y_test,
             n_qubits, noise_level, rng) -> dict:
    from src.quantum_models import VQCClassifier

    X_tr, y_tr = _subsample_stratified(X_train, y_train, VQC_TRAIN_SAMPLES, rng)
    X_te, y_te = _subsample_stratified(X_test,  y_test,  MAX_TEST_SAMPLES,  rng)

    clf = VQCClassifier(
        n_qubits=n_qubits,
        n_layers=2,
        n_epochs=VQC_EPOCHS,
        noise_level=noise_level,
        backend="default.qubit",   # default.mixed auto-selected when noise_level>0
    )
    clf.fit(X_tr, y_tr)

    # Threshold tuning on real-distribution val set
    val_prob  = clf.predict_proba(X_val)[:, 1]
    threshold = find_optimal_threshold(y_val, val_prob)
    y_prob    = clf.predict_proba(X_te)
    y_pred    = (y_prob[:, 1] >= threshold).astype(int)

    m = evaluate_model(f"VQC (p={noise_level})", y_te, y_pred, y_prob)
    return {"model": "VQC", "noise_level": noise_level, "metrics": m.to_dict()}


def _run_qsvm(X_train, y_train, X_test, y_test,
              n_qubits, noise_level, rng) -> dict:
    from src.quantum_models import QSVMClassifier

    X_tr, y_tr = _subsample_stratified(X_train, y_train, QSVM_TRAIN_SAMPLES, rng)
    X_te, y_te = _subsample_stratified(X_test,  y_test,  MAX_TEST_SAMPLES,   rng)

    clf = QSVMClassifier(
        n_qubits=n_qubits,
        noise_level=noise_level,
        backend="default.qubit",
    )
    clf.fit(X_tr, y_tr)

    y_prob = clf.predict_proba(X_te)
    y_pred = clf.predict(X_te)
    m = evaluate_model(f"QSVM (p={noise_level})", y_te, y_pred, y_prob)
    return {"model": "QSVM", "noise_level": noise_level, "metrics": m.to_dict()}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_noise_sweep(
    noise_levels: list[float],
    quantum_results: list[dict],
    classical_results: list[dict],
    save_path: Path,
) -> None:
    metrics_to_plot = ["f1_fraud", "mcc", "pr_auc", "roc_auc"]
    titles = {"f1_fraud": "F1-Fraud", "mcc": "MCC",
              "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    quantum_models = list(dict.fromkeys(r["model"] for r in quantum_results))
    colors = {"VQC": "#D65F5F", "QSVM": "#B47CC7"}
    classical_colors = {"Random Forest": "#4878CF", "XGBoost": "#6ACC65"}
    classical_styles = {"Random Forest": "--", "XGBoost": "--"}

    for ax, metric in zip(axes, metrics_to_plot):
        # Quantum model lines (one per model)
        for qm in quantum_models:
            pts = sorted(
                [r for r in quantum_results if r["model"] == qm],
                key=lambda r: r["noise_level"],
            )
            xs = [r["noise_level"] for r in pts]
            ys = [r["metrics"][metric] for r in pts]
            ax.plot(xs, ys, marker="o", linewidth=2,
                    color=colors.get(qm, "gray"), label=qm)

        # Classical baselines (horizontal dashed lines)
        for cr in classical_results:
            val = cr["metrics"][metric]
            ax.axhline(
                val, linestyle=classical_styles.get(cr["model"], "--"),
                color=classical_colors.get(cr["model"], "black"),
                linewidth=1.5, alpha=0.8,
                label=f"{cr['model']} (classical)",
            )

        # Shade NISQ-era range (p≈0.001–0.01)
        ax.axvspan(0.001, 0.01, alpha=0.08, color="orange",
                   label="NISQ range" if metric == "f1_fraud" else "_nolegend_")

        ax.set_title(titles[metric])
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if metric == "f1_fraud":
            ax.legend(fontsize=7, loc="lower left")

    for ax in axes:
        ax.set_xlabel("Depolarizing error probability p")

    fig.suptitle(
        "Quantum Model Performance vs Depolarizing Noise\n"
        "QML Fraud Detection Benchmark — Shaded region: current NISQ hardware",
        fontsize=12,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved %s", save_path)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _merge(args) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%H:%M:%S")
    all_results: list[dict] = []
    for d in args.noise_dirs:
        f = d / "noise_results.json"
        if not f.exists():
            logging.warning("Not found, skipping: %s", f)
            continue
        with open(f) as fh:
            all_results.extend(json.load(fh))
        logging.info("Loaded %s", f)

    # Deduplicate by (model, noise_level), keep last seen
    seen: dict[tuple, dict] = {}
    for r in all_results:
        seen[(r["model"], r["noise_level"])] = r
    merged = list(seen.values())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "noise_results.json"
    with open(out_json, "w") as fh:
        json.dump(merged, fh, indent=2)
    logging.info("Merged %d records → %s", len(merged), out_json)

    if not args.no_plots:
        classical = [r for r in merged if r["noise_level"] is None]
        quantum   = [r for r in merged if r["noise_level"] is not None]
        levels    = sorted(set(r["noise_level"] for r in quantum))
        _plot_noise_sweep(levels, quantum, classical, args.out_dir / "noise_vs_metric.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if args.command == "merge":
        _merge(args)
        return

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  QML Fraud Detection — Noise Sweep                  ║")
    print(f"║  Levels: {args.noise_levels}")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Preprocess ────────────────────────────────────────────────────────────
    logging.info("Preprocessing…")
    cfg  = PreprocessingConfig(data_path=args.data_path, n_qubits=args.n_qubits)
    data = preprocess(cfg)
    rng  = np.random.default_rng(42)

    # Subsample test set once (shared across all noise levels for fair comparison)
    X_te, y_te = _subsample_stratified(
        data.X_test, data.y_test, MAX_TEST_SAMPLES, rng
    )

    # ── Classical baselines (noise-independent) ───────────────────────────────
    logging.info("Evaluating classical baselines…")
    classical_results = _classical_baselines(MODELS_DIR, X_te, y_te)

    # ── Noise sweep ───────────────────────────────────────────────────────────
    quantum_results: list[dict] = []
    t_total = time.perf_counter()

    for p in args.noise_levels:
        logging.info("\n%s\n  noise_level = %.4f\n%s", "=" * 50, p, "=" * 50)

        t0 = time.perf_counter()
        res = _run_vqc(
            data.X_train, data.y_train,
            data.X_val,   data.y_val,
            data.X_test,  data.y_test,
            args.n_qubits, p, rng,
        )
        quantum_results.append(res)
        logging.info("VQC  p=%.4f | F1-fraud=%.4f MCC=%.4f | %.1f s",
                     p, res["metrics"]["f1_fraud"], res["metrics"]["mcc"],
                     time.perf_counter() - t0)

        if not args.vqc_only:
            t0 = time.perf_counter()
            res = _run_qsvm(
                data.X_train, data.y_train,
                data.X_test,  data.y_test,
                args.n_qubits, p, rng,
            )
            quantum_results.append(res)
            logging.info("QSVM p=%.4f | F1-fraud=%.4f MCC=%.4f | %.1f s",
                         p, res["metrics"]["f1_fraud"], res["metrics"]["mcc"],
                         time.perf_counter() - t0)

    logging.info("Total sweep time: %.1f min",
                 (time.perf_counter() - t_total) / 60)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    args.noise_dir.mkdir(parents=True, exist_ok=True)
    all_results = classical_results + quantum_results
    out_json = args.noise_dir / "noise_results.json"
    with open(out_json, "w") as fh:
        json.dump(all_results, fh, indent=2)
    logging.info("Results saved to %s", out_json)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\nNoise Sweep Summary — F1-Fraud\n")
    print(f"{'Model':<8}  " + "  ".join(f"p={p:.3f}" for p in args.noise_levels))
    print("-" * (10 + 10 * len(args.noise_levels)))

    for qm in (["VQC"] + ([] if args.vqc_only else ["QSVM"])):
        row = f"{qm:<8}  "
        for p in args.noise_levels:
            match = next(
                (r for r in quantum_results
                 if r["model"] == qm and r["noise_level"] == p), None
            )
            row += f"  {match['metrics']['f1_fraud']:.4f}" if match else "     N/A"
        print(row)

    print("\nClassical baselines (noise-independent):")
    for cr in classical_results:
        print(f"  {cr['model']:<20} F1-fraud={cr['metrics']['f1_fraud']:.4f}")
    print()

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not args.no_plots:
        _plot_noise_sweep(
            args.noise_levels, quantum_results, classical_results,
            args.noise_dir / "noise_vs_metric.png",
        )

    print(f"Noise sweep complete. Results saved to {args.noise_dir}/")


if __name__ == "__main__":
    main()
