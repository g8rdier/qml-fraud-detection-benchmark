"""
Unit tests for src/quantum_models.py.

Uses tiny toy datasets (4 features, 4 qubits) to keep circuit simulations fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.quantum_models import QSVMClassifier, VQCClassifier


N_QUBITS = 4
N_TRAIN = 20
N_TEST = 6
RNG = np.random.default_rng(99)

X_TRAIN = RNG.uniform(0, 1, size=(N_TRAIN, N_QUBITS))
Y_TRAIN = RNG.integers(0, 2, size=N_TRAIN)
X_TEST = RNG.uniform(0, 1, size=(N_TEST, N_QUBITS))


class TestVQCClassifier:
    @pytest.fixture(scope="class")
    def fitted_vqc(self):
        clf = VQCClassifier(
            n_qubits=N_QUBITS,
            n_layers=1,
            n_epochs=5,          # minimal epochs for speed
            backend="default.qubit",
        )
        clf.fit(X_TRAIN, Y_TRAIN)
        return clf

    def test_fit_sets_weights(self, fitted_vqc):
        assert hasattr(fitted_vqc, "weights_")
        assert hasattr(fitted_vqc, "normaliser_")

    def test_predict_shape(self, fitted_vqc):
        preds = fitted_vqc.predict(X_TEST)
        assert preds.shape == (N_TEST,)

    def test_predict_binary(self, fitted_vqc):
        preds = fitted_vqc.predict(X_TEST)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, fitted_vqc):
        proba = fitted_vqc.predict_proba(X_TEST)
        assert proba.shape == (N_TEST, 2)

    def test_predict_proba_sums_to_one(self, fitted_vqc):
        proba = fitted_vqc.predict_proba(X_TEST)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestQSVMClassifier:
    @pytest.fixture(scope="class")
    def fitted_qsvm(self):
        # Use only 8 training samples to keep kernel matrix computation fast
        clf = QSVMClassifier(n_qubits=N_QUBITS, backend="default.qubit")
        clf.fit(X_TRAIN[:8], Y_TRAIN[:8])
        return clf

    def test_predict_shape(self, fitted_qsvm):
        preds = fitted_qsvm.predict(X_TEST[:3])
        assert preds.shape == (3,)

    def test_predict_binary(self, fitted_qsvm):
        preds = fitted_qsvm.predict(X_TEST[:3])
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, fitted_qsvm):
        proba = fitted_qsvm.predict_proba(X_TEST[:3])
        assert proba.shape == (3, 2)
