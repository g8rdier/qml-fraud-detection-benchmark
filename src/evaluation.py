"""
evaluation.py
=============
Model evaluation utilities for the QML Benchmark.

Prioritises fraud-detection-relevant metrics:
  - F1-Score (macro and fraud class)
  - Precision-Recall AUC
  - Matthews Correlation Coefficient (MCC)
  - ROC-AUC

Simple accuracy is deliberately de-emphasised because it is misleading on
heavily imbalanced datasets (a model predicting all-negative achieves >99%).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Return the probability threshold that maximises F1-fraud on a validation set.

    Uses the full precision-recall curve so no grid search is needed.
    Should be called on a *validation* set, never the final test set.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds has one fewer element than precisions/recalls
    f1s = np.where(
        (precisions[:-1] + recalls[:-1]) == 0,
        0.0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
    )
    best_idx = int(np.argmax(f1s))
    best_threshold = float(thresholds[best_idx])
    logger.info(
        "Optimal threshold: %.4f  →  val precision=%.4f  recall=%.4f  F1=%.4f",
        best_threshold, precisions[best_idx], recalls[best_idx], f1s[best_idx],
    )
    return best_threshold


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    model_name: str
    f1_macro: float
    f1_fraud: float           # F1 for the minority (fraud) class
    pr_auc: float             # Area under Precision-Recall curve
    roc_auc: float
    mcc: float                # Matthews Correlation Coefficient
    accuracy: float           # Reported last, as a sanity-check only

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"[{self.model_name}]\n"
            f"  F1 (macro)  : {self.f1_macro:.4f}\n"
            f"  F1 (fraud)  : {self.f1_fraud:.4f}\n"
            f"  PR-AUC      : {self.pr_auc:.4f}\n"
            f"  ROC-AUC     : {self.roc_auc:.4f}\n"
            f"  MCC         : {self.mcc:.4f}\n"
            f"  Accuracy    : {self.accuracy:.4f}  (reference only)\n"
        )


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> ModelMetrics:
    """
    Compute all benchmark metrics for a single model.

    Parameters
    ----------
    model_name : str
        Human-readable label (e.g. "XGBoost", "VQC").
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels.
    y_pred : array-like of shape (n_samples,)
        Hard binary predictions.
    y_prob : array-like of shape (n_samples,) or (n_samples, 2), optional
        Probability scores for the positive class.  Required for ROC-AUC
        and PR-AUC; set to ``None`` to skip those metrics.
    """
    if y_prob is not None and y_prob.ndim == 2:
        y_prob = y_prob[:, 1]

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_fraud = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    acc = (y_true == y_pred).mean()

    if y_prob is not None:
        pr_auc = average_precision_score(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
    else:
        pr_auc = roc_auc = float("nan")
        logger.warning("%s: no probability scores — PR-AUC/ROC-AUC unavailable.",
                       model_name)

    metrics = ModelMetrics(
        model_name=model_name,
        f1_macro=f1_macro,
        f1_fraud=f1_fraud,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        mcc=mcc,
        accuracy=acc,
    )
    logger.info("\n%s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Comparison & reporting
# ---------------------------------------------------------------------------

def compare_models(results: list[ModelMetrics]) -> None:
    """Print a side-by-side comparison table."""
    header = f"{'Model':<20} {'F1-Macro':>9} {'F1-Fraud':>9} {'PR-AUC':>8} {'ROC-AUC':>8} {'MCC':>8}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r.model_name:<20} "
            f"{r.f1_macro:>9.4f} "
            f"{r.f1_fraud:>9.4f} "
            f"{r.pr_auc:>8.4f} "
            f"{r.roc_auc:>8.4f} "
            f"{r.mcc:>8.4f}"
        )
    print(sep)


def save_metrics_json(results: list[ModelMetrics], path: str | Path) -> None:
    """Persist all metrics to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump([r.to_dict() for r in results], fh, indent=2)
    logger.info("Metrics saved to %s", path)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_pr_curves(
    models: list[dict],
    save_path: str | Path | None = None,
) -> None:
    """
    Overlay Precision-Recall curves for multiple models.

    Parameters
    ----------
    models : list of dict
        Each dict: ``{"name": str, "y_true": array, "y_prob": array}``.
    save_path : path, optional
        If provided, save figure instead of displaying it.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in models:
        PrecisionRecallDisplay.from_predictions(
            m["y_true"], m["y_prob"], name=m["name"], ax=ax
        )
    ax.set_title("Precision-Recall Curves — Fraud Detection Benchmark")
    ax.legend(loc="upper right")
    _save_or_show(fig, save_path)


def plot_roc_curves(
    models: list[dict],
    save_path: str | Path | None = None,
) -> None:
    """Overlay ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in models:
        RocCurveDisplay.from_predictions(
            m["y_true"], m["y_prob"], name=m["name"], ax=ax
        )
    ax.set_title("ROC Curves — Fraud Detection Benchmark")
    ax.legend(loc="lower right")
    _save_or_show(fig, save_path)


def plot_confusion_matrices(
    models: list[dict],
    save_path: str | Path | None = None,
    normalize: bool = False,
) -> None:
    """
    Plot confusion matrices for all models in a grid.

    Parameters
    ----------
    models : list of dict
        Each dict: ``{"name": str, "y_true": array, "y_pred": array}``.
    save_path : path, optional
        If provided, save figure instead of displaying it.
    normalize : bool, optional
        If True, display normalized (percentage) confusion matrices.
        If False (default), display raw counts.
    """
    n = len(models)
    # Use 2x2 grid for better readability
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, m in zip(axes, models):
        cm = confusion_matrix(m["y_true"], m["y_pred"])
        n_samples = len(m["y_true"])

        # Optionally normalize to percentages
        if normalize:
            cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        else:
            cm_display = cm

        # Create display with appropriate values
        display = ConfusionMatrixDisplay(cm_display, display_labels=["Legit", "Fraud"])
        display.plot(ax=ax, colorbar=False, cmap='YlOrRd')

        # Add sample size and model name to title
        title = f"{m['name']}\n(n={n_samples:,})"
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Adjust text formatting for normalized matrices
        if normalize:
            ax.set_xlabel("Predicted label (%)")
            ax.set_ylabel("True label (%)")

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "Confusion Matrices" + (" — Normalized (%)" if normalize else ""),
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_confusion_matrices_quantum_focus(
    models: list[dict],
    save_path: str | Path | None = None,
    normalize: bool = False,
) -> None:
    """
    Plot confusion matrices for 3 models (XGBoost, VQC, QSVM) side-by-side.
    Excludes Random Forest for a focused quantum vs best-classical comparison.

    Parameters
    ----------
    models : list of dict
        Each dict: ``{"name": str, "y_true": array, "y_pred": array}``.
    save_path : path, optional
        If provided, save figure instead of displaying it.
    normalize : bool, optional
        If True, display normalized (percentage) confusion matrices.
    """
    # Filter to only include XGBoost, VQC, QSVM
    filtered = [m for m in models if any(x in m["name"] for x in ["XGBoost", "VQC", "QSVM"])]

    n = len(filtered)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, m in zip(axes, filtered):
        cm = confusion_matrix(m["y_true"], m["y_pred"])
        n_samples = len(m["y_true"])

        # Optionally normalize to percentages
        if normalize:
            cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        else:
            cm_display = cm

        # Create display with appropriate values
        display = ConfusionMatrixDisplay(cm_display, display_labels=["Legit", "Fraud"])
        display.plot(ax=ax, colorbar=False, cmap='YlOrRd')

        # Add sample size and model name to title
        title = f"{m['name']}\n(n={n_samples:,})"
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Adjust text formatting for normalized matrices
        if normalize:
            ax.set_xlabel("Predicted label (%)")
            ax.set_ylabel("True label (%)")

    fig.suptitle(
        "Confusion Matrices — Quantum vs Classical" + (" — Normalized (%)" if normalize else ""),
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_metric_comparison(
    results: list[ModelMetrics],
    save_path: str | Path | None = None,
) -> None:
    """Bar chart comparing key metrics across models."""
    metrics_of_interest = ["f1_fraud", "pr_auc", "roc_auc", "mcc"]
    labels = [r.model_name for r in results]
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics_of_interest):
        values = [getattr(r, metric) for r in results]
        ax.bar(x + i * width, values, width, label=metric.upper())

    ax.set_xticks(x + width * (len(metrics_of_interest) - 1) / 2)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Key Fraud-Detection Metrics")
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_calibration_curves(
    models: list[dict],
    save_path: str | Path | None = None,
) -> None:
    """
    Score distribution plot: P(fraud) histograms for legit vs fraud transactions.

    This is more informative than a reliability diagram for heavily imbalanced
    data.  It directly shows the SMOTE miscalibration effect: models trained on
    balanced synthetic data push fraud probabilities toward 1.0, meaning the
    optimal decision threshold is far above the naive 0.5.

    Parameters
    ----------
    models : list of dict
        Each dict: ``{"name": str, "y_true": array, "y_prob": array}``.
    save_path : path, optional
        If provided, save figure instead of displaying it.
    """
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, m in zip(axes, models):
        y_true = np.asarray(m["y_true"])
        y_prob = np.asarray(m["y_prob"])

        prob_legit = y_prob[y_true == 0]
        prob_fraud = y_prob[y_true == 1]

        bins = np.linspace(0, 1, 60)
        ax.hist(prob_legit, bins=bins, alpha=0.6, color=colors[0],
                label=f"Legit (n={len(prob_legit):,})", density=True)
        ax.hist(prob_fraud, bins=bins, alpha=0.7, color=colors[1],
                label=f"Fraud (n={len(prob_fraud):,})", density=True)

        # Mark the decision threshold if stored in the model name (τ=...)
        import re
        match = re.search(r"τ=([0-9.]+)", m["name"])
        if match:
            tau = float(match.group(1))
            ax.axvline(tau, color="red", linestyle="--", lw=1.5,
                       label=f"Threshold τ={tau:.3f}")
        else:
            ax.axvline(0.5, color="red", linestyle="--", lw=1.5,
                       label="Threshold τ=0.500")

        ax.set_xlabel("P(fraud) — model output")
        ax.set_ylabel("Density")
        ax.set_title(m["name"])
        ax.legend(fontsize=8)

    fig.suptitle(
        "Score Distributions — SMOTE Probability Calibration Effect\n"
        "Fraud scores pushed high by balanced training; threshold must compensate",
        fontsize=11,
    )
    plt.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig: plt.Figure, path: str | Path | None) -> None:
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved to %s", path)
        plt.close(fig)
    else:
        plt.show()
