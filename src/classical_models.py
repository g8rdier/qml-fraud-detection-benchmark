"""
classical_models.py
===================
Classical baseline models: Random Forest and XGBoost.

Includes builder functions and a unified `train_classical_model()` entry
point that handles fitting, timing, cross-validation, and model persistence.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------

def build_random_forest(
    n_estimators: int = 300,
    max_depth: int | None = None,
    class_weight: dict | str | None = "balanced",
    random_state: int = 42,
) -> RandomForestClassifier:
    """Return a configured Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=random_state,
    )


def build_xgboost(
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    scale_pos_weight: float = 1.0,
    random_state: int = 42,
) -> XGBClassifier:
    """
    Return a configured XGBoost classifier.

    Parameters
    ----------
    scale_pos_weight : float
        Set to ``n_negative / n_positive`` when SMOTE is NOT used.
        Ignored (left at 1.0) when training data is already balanced by SMOTE.
    """
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_classical_model(
    model: RandomForestClassifier | XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    cv_folds: int = 5,
    save_dir: str | Path | None = None,
    model_name: str = "model",
    tune_threshold: bool = False,
) -> dict[str, Any]:
    """
    Fit a classical model, optionally cross-validate, and return predictions.

    Parameters
    ----------
    model : sklearn-compatible estimator
    X_train, y_train : training data (post-SMOTE)
    X_test, y_test : held-out test data
    X_val, y_val : pre-SMOTE validation set at real class distribution.
        Required when tune_threshold=True.
    cv_folds : int
        Number of stratified CV folds.  Set to 0 to skip CV.
    save_dir : path, optional
        Directory to persist the fitted model as a joblib file.
    model_name : str
        Label used for logging and file naming.
    tune_threshold : bool
        If True, use X_val/y_val (real class distribution, pre-SMOTE) to find
        the F1-fraud-maximising decision threshold and apply it to test
        predictions.  X_val/y_val must be provided.

    Returns
    -------
    dict with keys:
        ``y_pred``      – hard predictions on X_test (threshold-tuned if requested)
        ``y_prob``      – probability array of shape (n_test, 2)
        ``fit_time``    – wall-clock seconds for fit()
        ``cv_scores``   – dict of CV metric arrays (empty if cv_folds=0)
        ``threshold``   – decision threshold used (0.5 default or tuned value)
    """
    from src.evaluation import find_optimal_threshold

    logger.info("=== Training %s ===", model_name)

    # ── Optional threshold tuning on pre-SMOTE validation set ──────────────
    threshold = 0.5
    if tune_threshold:
        if X_val is None or y_val is None:
            raise ValueError("tune_threshold=True requires X_val and y_val.")
        logger.info(
            "Threshold tuning: pre-fitting on train, validating on %d real samples…",
            len(X_val),
        )
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        threshold = find_optimal_threshold(y_val, val_prob)
        logger.info("Tuned threshold: %.4f (was 0.5000)", threshold)

    # ── Cross-validation (on full training data) ───────────────────────────
    cv_scores: dict[str, np.ndarray] = {}
    if cv_folds > 0:
        logger.info("Running %d-fold stratified cross-validation…", cv_folds)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        raw = cross_validate(
            model, X_train, y_train,
            cv=cv,
            scoring={
                "f1":        "f1",
                "roc_auc":   "roc_auc",
                "precision": "precision",
                "recall":    "recall",
            },
            return_train_score=False,
            n_jobs=-1,
        )
        cv_scores = {k.replace("test_", ""): v for k, v in raw.items()
                     if k.startswith("test_")}
        for metric, values in cv_scores.items():
            logger.info(
                "  CV %-12s: %.4f ± %.4f", metric, values.mean(), values.std()
            )

    # ── Final fit on full training set ─────────────────────────────────────
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    logger.info("Fit completed in %.2f s", fit_time)

    # ── Predict on held-out test set ───────────────────────────────────────
    y_prob = model.predict_proba(X_test)
    y_pred = (y_prob[:, 1] >= threshold).astype(int)

    # ── Persist model ──────────────────────────────────────────────────────
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{model_name}.joblib"
        joblib.dump(model, out_path)
        logger.info("Model saved to %s", out_path)

    return {
        "y_pred":    y_pred,
        "y_prob":    y_prob,
        "fit_time":  fit_time,
        "cv_scores": cv_scores,
        "threshold": threshold,
    }
