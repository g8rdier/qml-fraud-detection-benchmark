"""
Tests for run_latency.py.

Runs classical-only latency analysis on a tiny synthetic dataset to verify
JSON output structure and that the summary table is printed without errors.
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
    rng = np.random.default_rng(1)
    n = 600
    X = rng.standard_normal((n, 10))
    y = np.zeros(n, dtype=int)
    y[:20] = 1
    rng.shuffle(y)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 11)])
    df["Amount"] = rng.exponential(100, n)
    df["Time"]   = np.arange(n)
    df["Class"]  = y
    path = tmp_path_factory.mktemp("data") / "creditcard.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_latency_classical_only(synthetic_csv, tmp_path, monkeypatch):
    """Classical-only latency run: produces valid JSON with timing stats."""
    monkeypatch.chdir(tmp_path)

    # Need saved models — run benchmark first to produce joblib files
    bench = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent.parent / "run_benchmark.py"),
            "--data-path", synthetic_csv,
            "--classical-only",
            "--cv-folds", "0",
            "--no-plots",
            "--n-qubits", "2",
        ],
        capture_output=True, text=True,
    )
    assert bench.returncode == 0, bench.stderr

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent.parent / "run_latency.py"),
            "--data-path",    synthetic_csv,
            "--classical-only",
            "--n-repeats",    "20",
            "--n-qubits",     "2",
            "--no-plots",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    out_json = tmp_path / "results" / "latency" / "latency_results.json"
    assert out_json.exists(), "latency_results.json not created"

    data = json.loads(out_json.read_text())
    assert len(data) == 2  # RF + XGBoost

    for entry in data:
        assert "model"   in entry
        assert "latency" in entry
        s = entry["latency"]
        assert s["median_ms"] > 0
        assert s["p99_ms"] >= s["median_ms"]
        assert s["n_samples"] == 20
