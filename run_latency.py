"""
run_latency.py
==============
Per-sample inference latency analysis for all four benchmark models.

Measures single-sample predict_proba latency (median, IQR, p99) to answer
a critical production question: can quantum models classify transactions
fast enough for real-time fraud detection?

Models benchmarked
------------------
- Random Forest  — loaded from results/models/random_forest.joblib
- XGBoost        — loaded from results/models/xgboost.joblib
- VQC            — trained with minimal epochs (latency is epoch-independent)
- QSVM           — trained on small n_train; latency scales linearly with n_train

Outputs
-------
results/latency/latency_results.json   — timing stats per model
results/latency/latency_comparison.png — bar chart (log scale)

Usage
-----
python run_latency.py                          # full (classical + quantum)
python run_latency.py --classical-only         # fast, skips quantum training
python run_latency.py --n-repeats 200          # more timing samples
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

from src.preprocessing import PreprocessingConfig, preprocess

LATENCY_DIR = Path("results/latency")
MODELS_DIR  = Path("results/models")
DATA_PATH   = Path("data/raw/creditcard.csv")

N_WARMUP   = 5    # discard first N calls (JIT / cache warm-up)
N_REPEATS  = 100  # timing samples per model

# Minimal quantum training settings — only affect fit time, not predict latency
VQC_TRAIN_SAMPLES  = 100
VQC_EPOCHS         = 5
QSVM_TRAIN_SAMPLES = 50   # latency scales with this; see extrapolation note


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-sample inference latency analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-path",       type=Path, default=DATA_PATH)
    p.add_argument("--n-qubits",        type=int,  default=8)
    p.add_argument("--quantum-backend", type=str,  default="lightning.qubit")
    p.add_argument("--n-repeats",       type=int,  default=N_REPEATS)
    p.add_argument("--classical-only",  action="store_true")
    p.add_argument("--no-plots",        action="store_true")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_single_sample(
    predict_fn,
    X: np.ndarray,
    n_warmup: int,
    n_repeats: int,
) -> dict:
    """
    Time predict_fn(x) for single samples drawn from X.

    Returns dict with keys: median_ms, mean_ms, std_ms, p25_ms, p75_ms, p99_ms.
    """
    rng = np.random.default_rng(0)
    indices = rng.integers(0, len(X), size=n_warmup + n_repeats)

    # Warm-up — JIT compilation, cache effects
    for i in indices[:n_warmup]:
        predict_fn(X[i : i + 1])

    # Timed runs
    times_ms = []
    for i in indices[n_warmup:]:
        t0 = time.perf_counter()
        predict_fn(X[i : i + 1])
        times_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times_ms)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms":   float(arr.mean()),
        "std_ms":    float(arr.std()),
        "p25_ms":    float(np.percentile(arr, 25)),
        "p75_ms":    float(np.percentile(arr, 75)),
        "p99_ms":    float(np.percentile(arr, 99)),
        "n_samples": n_repeats,
    }


# ---------------------------------------------------------------------------
# Model loading / training
# ---------------------------------------------------------------------------

def _load_classical(models_dir: Path) -> dict:
    """Load saved RF and XGBoost from joblib."""
    rf_path  = models_dir / "random_forest.joblib"
    xgb_path = models_dir / "xgboost.joblib"

    if not rf_path.exists() or not xgb_path.exists():
        raise FileNotFoundError(
            f"Saved models not found in {models_dir}. "
            "Run run_benchmark.py first."
        )
    return {
        "Random Forest": joblib.load(rf_path),
        "XGBoost":       joblib.load(xgb_path),
    }


def _train_vqc_for_timing(X_train, y_train, n_qubits, backend) -> object:
    """Fit a VQC with minimal epochs — only weights matter for latency measurement."""
    from src.quantum_models import VQCClassifier
    clf = VQCClassifier(
        n_qubits=n_qubits,
        n_layers=2,
        n_epochs=VQC_EPOCHS,
        backend=backend,
    )
    # Stratified subsample
    rng = np.random.default_rng(42)
    idx0 = np.where(y_train == 0)[0]
    idx1 = np.where(y_train == 1)[0]
    half = VQC_TRAIN_SAMPLES // 2
    idx = np.concatenate([
        rng.choice(idx0, size=min(half, len(idx0)), replace=False),
        rng.choice(idx1, size=min(half, len(idx1)), replace=False),
    ])
    clf.fit(X_train[idx], y_train[idx])
    return clf


def _train_qsvm_for_timing(X_train, y_train, n_qubits, backend) -> object:
    """Fit a QSVM on QSVM_TRAIN_SAMPLES samples for latency measurement."""
    from src.quantum_models import QSVMClassifier
    clf = QSVMClassifier(n_qubits=n_qubits, backend=backend)
    rng = np.random.default_rng(42)
    idx0 = np.where(y_train == 0)[0]
    idx1 = np.where(y_train == 1)[0]
    half = QSVM_TRAIN_SAMPLES // 2
    idx = np.concatenate([
        rng.choice(idx0, size=min(half, len(idx0)), replace=False),
        rng.choice(idx1, size=min(half, len(idx1)), replace=False),
    ])
    clf.fit(X_train[idx], y_train[idx])
    return clf


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_latency(results: list[dict], save_path: Path) -> None:
    names   = [r["model"] for r in results]
    medians = [r["latency"]["median_ms"] for r in results]
    p25     = [r["latency"]["p25_ms"]    for r in results]
    p75     = [r["latency"]["p75_ms"]    for r in results]

    err_lo = [m - p for m, p in zip(medians, p25)]
    err_hi = [p - m for m, p in zip(medians, p75)]

    colors = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, medians, color=colors[:len(names)], width=0.5,
                  yerr=[err_lo, err_hi], capsize=5, error_kw={"linewidth": 1.5})

    # Annotate bars with median value
    for bar, val in zip(bars, medians):
        label = f"{val:.2f} ms" if val >= 1 else f"{val*1000:.1f} µs"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Latency per sample — log scale (ms)")
    ax.set_title(
        "Single-Sample Inference Latency\n"
        "Median ± IQR (log scale) — QML Fraud Detection Benchmark"
    )
    ax.axhline(10,   color="orange", linestyle="--", linewidth=1, alpha=0.7,
               label="10 ms (100 tx/s)")
    ax.axhline(100,  color="red",    linestyle="--", linewidth=1, alpha=0.7,
               label="100 ms (10 tx/s)")
    ax.axhline(1000, color="darkred", linestyle="--", linewidth=1, alpha=0.5,
               label="1 s (1 tx/s)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
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
    print("║  QML Fraud Detection — Latency Analysis              ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Preprocess ────────────────────────────────────────────────────────────
    logging.info("Preprocessing data…")
    cfg  = PreprocessingConfig(data_path=args.data_path, n_qubits=args.n_qubits)
    data = preprocess(cfg)
    X_test = data.X_test

    all_results: list[dict] = []

    # ── Classical models ──────────────────────────────────────────────────────
    logging.info("Loading classical models from %s…", MODELS_DIR)
    classical = _load_classical(MODELS_DIR)

    for name, clf in classical.items():
        logging.info("Timing %s (%d repeats)…", name, args.n_repeats)
        stats = _time_single_sample(clf.predict_proba, X_test, N_WARMUP, args.n_repeats)
        all_results.append({"model": name, "latency": stats, "notes": ""})
        logging.info("  %s median=%.3f ms  p99=%.3f ms", name,
                     stats["median_ms"], stats["p99_ms"])

    # ── Quantum models ────────────────────────────────────────────────────────
    if not args.classical_only:
        # VQC
        logging.info(
            "Training VQC for timing (%d samples, %d epochs)…",
            VQC_TRAIN_SAMPLES, VQC_EPOCHS,
        )
        vqc = _train_vqc_for_timing(
            data.X_train, data.y_train, args.n_qubits, args.quantum_backend
        )
        logging.info("Timing VQC (%d repeats)…", args.n_repeats)
        vqc_stats = _time_single_sample(vqc.predict_proba, X_test, N_WARMUP, args.n_repeats)
        all_results.append({
            "model":   "VQC",
            "latency": vqc_stats,
            "notes":   f"{args.n_qubits} qubits, 2 layers, {args.quantum_backend}",
        })
        logging.info("  VQC median=%.3f ms  p99=%.3f ms",
                     vqc_stats["median_ms"], vqc_stats["p99_ms"])

        # QSVM
        logging.info(
            "Training QSVM for timing (%d train samples)…", QSVM_TRAIN_SAMPLES
        )
        qsvm = _train_qsvm_for_timing(
            data.X_train, data.y_train, args.n_qubits, args.quantum_backend
        )
        logging.info("Timing QSVM (%d repeats)…", args.n_repeats)
        qsvm_stats = _time_single_sample(qsvm.predict_proba, X_test, N_WARMUP, args.n_repeats)
        # Extrapolate to benchmark n_train=300
        scale = 300 / QSVM_TRAIN_SAMPLES
        qsvm_extrapolated_ms = qsvm_stats["median_ms"] * scale
        all_results.append({
            "model":   "QSVM",
            "latency": qsvm_stats,
            "notes": (
                f"Timed on n_train={QSVM_TRAIN_SAMPLES}; "
                f"benchmark n_train=300 extrapolates to ~{qsvm_extrapolated_ms:.0f} ms/sample"
            ),
        })
        logging.info(
            "  QSVM median=%.3f ms  p99=%.3f ms  (n_train=%d; extrapolated to n_train=300: ~%.0f ms)",
            qsvm_stats["median_ms"], qsvm_stats["p99_ms"],
            QSVM_TRAIN_SAMPLES, qsvm_extrapolated_ms,
        )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    LATENCY_DIR.mkdir(parents=True, exist_ok=True)
    out_json = LATENCY_DIR / "latency_results.json"
    with open(out_json, "w") as fh:
        json.dump(all_results, fh, indent=2)
    logging.info("Results saved to %s", out_json)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\nLatency Summary\n")
    header = f"{'Model':<20} {'Median (ms)':>12} {'p25 (ms)':>10} {'p75 (ms)':>10} {'p99 (ms)':>10}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        s = r["latency"]
        print(f"{r['model']:<20} {s['median_ms']:>12.3f} {s['p25_ms']:>10.3f} "
              f"{s['p75_ms']:>10.3f} {s['p99_ms']:>10.3f}")
        if r["notes"]:
            print(f"  {'':20} ↳ {r['notes']}")
    print()

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not args.no_plots:
        _plot_latency(all_results, LATENCY_DIR / "latency_comparison.png")

    print(f"Latency analysis complete. Results saved to {LATENCY_DIR}/")


if __name__ == "__main__":
    main()
