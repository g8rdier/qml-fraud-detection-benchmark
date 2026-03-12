"""
generate_plots.py
=================
Regenerate all benchmark figures from saved predictions JSON.

Decouples figure generation from model training: run once with
``--save-predictions``, then regenerate figures at any time without
re-running the (slow) quantum models.

Usage
-----
# After a benchmark run with --save-predictions:
python generate_plots.py --predictions results/metrics/predictions_<timestamp>.json

# Custom output directory:
python generate_plots.py --predictions results/metrics/predictions_<timestamp>.json \\
                         --out-dir results/figures
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.evaluation import (
    ModelMetrics,
    plot_calibration_curves,
    plot_confusion_matrices,
    plot_metric_comparison,
    plot_pr_curves,
    plot_roc_curves,
)

FIGURES_DIR = Path("results/figures")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regenerate benchmark figures from saved predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--predictions", type=Path, required=True,
        help="Path to predictions_<timestamp>.json produced by run_benchmark.py --save-predictions",
    )
    p.add_argument(
        "--out-dir", type=Path, default=FIGURES_DIR,
        help="Directory to write PNG figures into.",
    )
    p.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.predictions.exists():
        logging.error("Predictions file not found: %s", args.predictions)
        raise SystemExit(1)

    with open(args.predictions) as fh:
        raw = json.load(fh)

    plot_data = [
        {
            "name":   m["name"],
            "y_true": np.array(m["y_true"]),
            "y_pred": np.array(m["y_pred"]),
            "y_prob": np.array(m["y_prob"]),
        }
        for m in raw
    ]

    # Reconstruct lightweight ModelMetrics for the bar-chart
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        matthews_corrcoef,
        roc_auc_score,
    )

    results: list[ModelMetrics] = []
    for m in plot_data:
        y_true, y_pred, y_prob = m["y_true"], m["y_pred"], m["y_prob"]
        results.append(ModelMetrics(
            model_name=m["name"],
            f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
            f1_fraud=f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            pr_auc=average_precision_score(y_true, y_prob),
            roc_auc=roc_auc_score(y_true, y_prob),
            mcc=matthews_corrcoef(y_true, y_pred),
            accuracy=float((y_true == y_pred).mean()),
        ))

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    logging.info("Generating figures → %s", out)
    plot_pr_curves(plot_data,          save_path=out / "pr_curves.png")
    plot_roc_curves(plot_data,         save_path=out / "roc_curves.png")
    plot_confusion_matrices(plot_data, save_path=out / "confusion_matrices.png")
    plot_metric_comparison(results,    save_path=out / "metric_comparison.png")
    plot_calibration_curves(plot_data, save_path=out / "calibration_curves.png")

    print(f"\nAll figures saved to {out}/")


if __name__ == "__main__":
    main()
