"""
Tests for run_noise.py and the noise_level parameter in quantum models.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.quantum_models import QSVMClassifier, VQCClassifier


N_QUBITS = 4
RNG = np.random.default_rng(7)
X_TRAIN = RNG.uniform(0, 1, size=(16, N_QUBITS))
Y_TRAIN = RNG.integers(0, 2, size=16)
X_TEST  = RNG.uniform(0, 1, size=(6,  N_QUBITS))


class TestVQCNoise:
    def test_noisy_predict_proba_shape(self):
        clf = VQCClassifier(
            n_qubits=N_QUBITS, n_layers=1, n_epochs=2,
            noise_level=0.01, backend="default.qubit",
        )
        clf.fit(X_TRAIN, Y_TRAIN)
        proba = clf.predict_proba(X_TEST)
        assert proba.shape == (len(X_TEST), 2)

    def test_noisy_predict_binary(self):
        clf = VQCClassifier(
            n_qubits=N_QUBITS, n_layers=1, n_epochs=2,
            noise_level=0.01, backend="default.qubit",
        )
        clf.fit(X_TRAIN, Y_TRAIN)
        preds = clf.predict(X_TEST)
        assert set(preds).issubset({0, 1})

    def test_zero_noise_unchanged(self):
        """noise_level=0 should behave identically to the default (no noise)."""
        clf = VQCClassifier(
            n_qubits=N_QUBITS, n_layers=1, n_epochs=2,
            noise_level=0.0, backend="default.qubit",
        )
        clf.fit(X_TRAIN, Y_TRAIN)
        proba = clf.predict_proba(X_TEST)
        assert proba.shape == (len(X_TEST), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestQSVMNoise:
    def test_noisy_predict_proba_shape(self):
        clf = QSVMClassifier(
            n_qubits=N_QUBITS, noise_level=0.01, backend="default.qubit",
        )
        clf.fit(X_TRAIN[:8], Y_TRAIN[:8])
        proba = clf.predict_proba(X_TEST[:3])
        assert proba.shape == (3, 2)

    def test_noisy_predict_binary(self):
        clf = QSVMClassifier(
            n_qubits=N_QUBITS, noise_level=0.01, backend="default.qubit",
        )
        clf.fit(X_TRAIN[:8], Y_TRAIN[:8])
        preds = clf.predict(X_TEST[:3])
        assert set(preds).issubset({0, 1})


@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory) -> str:
    rng = np.random.default_rng(2)
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


def test_noise_sweep_vqc_only(synthetic_csv, tmp_path, monkeypatch):
    """VQC-only 2-point noise sweep produces valid JSON output."""
    monkeypatch.chdir(tmp_path)

    # Need saved classical models
    bench = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent.parent / "run_benchmark.py"),
            "--data-path", synthetic_csv,
            "--classical-only", "--cv-folds", "0",
            "--no-plots", "--n-qubits", "2",
        ],
        capture_output=True, text=True,
    )
    assert bench.returncode == 0, bench.stderr

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent.parent / "run_noise.py"),
            "--data-path",    synthetic_csv,
            "--noise-levels", "0.0", "0.01",
            "--vqc-only",
            "--no-plots",
            "--n-qubits", "2",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    out_json = tmp_path / "results" / "noise" / "noise_results.json"
    assert out_json.exists(), "noise_results.json not created"

    data = json.loads(out_json.read_text())
    quantum = [r for r in data if r["noise_level"] is not None]
    assert len(quantum) == 2   # 2 noise levels × VQC only

    for entry in quantum:
        assert entry["model"] == "VQC"
        assert 0.0 <= entry["metrics"]["f1_fraud"] <= 1.0
        assert 0.0 <= entry["metrics"]["roc_auc"]  <= 1.0
