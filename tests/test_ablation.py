"""
Unit / integration test for run_ablation.py.

Runs a 2-point sweep (n_qubits=2,4) on a tiny synthetic dataset with
--classical-only to keep runtime under a few seconds.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory) -> str:
    rng = np.random.default_rng(0)
    n = 600
    X = rng.standard_normal((n, 10))
    y = np.zeros(n, dtype=int)
    y[:20] = 1   # ~3.3% fraud — ensures ≥6 fraud in train after splits for SMOTE k=5
    rng.shuffle(y)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 11)])
    df["Amount"] = rng.exponential(100, n)
    df["Time"]   = np.arange(n)
    df["Class"]  = y
    path = tmp_path_factory.mktemp("data") / "creditcard.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_ablation_classical_only(synthetic_csv, tmp_path, monkeypatch):
    """Full script smoke-test: 2-point sweep, classical only, produces JSON + plots."""
    monkeypatch.chdir(tmp_path)

    result = subprocess.run(
        [
            sys.executable, str(Path(__file__).parent.parent / "run_ablation.py"),
            "--data-path", synthetic_csv,
            "--qubit-sweep", "2", "4",
            "--classical-only",
            "--cv-folds", "0",
            "--no-plots",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    out_json = tmp_path / "results" / "ablation" / "ablation_results.json"
    assert out_json.exists(), "ablation_results.json not created"

    data = json.loads(out_json.read_text())
    assert len(data) == 2, "Expected 2 sweep points"

    for point in data:
        assert "n_qubits"     in point
        assert "pca_variance" in point
        assert "metrics"      in point
        assert len(point["metrics"]) == 2   # RF + XGBoost
        for m in point["metrics"]:
            assert 0.0 <= m["f1_fraud"] <= 1.0
            assert 0.0 <= m["roc_auc"]  <= 1.0
