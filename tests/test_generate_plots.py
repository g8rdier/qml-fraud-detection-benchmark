"""
Tests for generate_plots.py and the --save-predictions flag in run_benchmark.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def predictions_json(tmp_path_factory) -> Path:
    """Write a minimal predictions JSON matching the format from --save-predictions."""
    rng = np.random.default_rng(0)
    n = 200
    y_true = np.zeros(n, dtype=int)
    y_true[:10] = 1
    rng.shuffle(y_true)

    models = []
    for name in ["Random Forest", "XGBoost (tuned τ=0.900)", "QSVM"]:
        y_prob = rng.uniform(0, 1, n)
        y_pred = (y_prob >= 0.5).astype(int)
        models.append({
            "name":   name,
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist(),
        })

    path = tmp_path_factory.mktemp("preds") / "predictions_test.json"
    path.write_text(json.dumps(models))
    return path


def test_generate_plots_creates_figures(predictions_json, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent.parent / "generate_plots.py"),
            "--predictions", str(predictions_json),
            "--out-dir",     str(tmp_path / "figures"),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    expected = [
        "pr_curves.png",
        "roc_curves.png",
        "confusion_matrices.png",
        "metric_comparison.png",
        "calibration_curves.png",
    ]
    for fname in expected:
        assert (tmp_path / "figures" / fname).exists(), f"{fname} not created"
