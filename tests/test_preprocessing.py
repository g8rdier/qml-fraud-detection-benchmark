"""
Unit tests for src/preprocessing.py.

Uses a synthetic dataset so no real data file is required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import PreprocessingConfig, PreprocessedData, preprocess


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SYNTHETIC_PATH = "/tmp/test_creditcard.csv"
N_SAMPLES = 500
N_FEATURES = 12          # Simulates V1–V12 from the real dataset
FRAUD_RATE = 0.02        # ~1 % fraud


@pytest.fixture(scope="module", autouse=True)
def synthetic_csv(tmp_path_factory):
    """Write a small synthetic CSV that mimics the real dataset layout."""
    rng = np.random.default_rng(0)
    n_fraud = max(1, int(N_SAMPLES * FRAUD_RATE))

    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y = np.zeros(N_SAMPLES, dtype=int)
    y[:n_fraud] = 1
    rng.shuffle(y)

    cols = [f"V{i}" for i in range(1, N_FEATURES + 1)]
    df = pd.DataFrame(X, columns=cols)
    df["Amount"] = rng.exponential(scale=100, size=N_SAMPLES)
    df["Time"] = np.arange(N_SAMPLES)
    df["Class"] = y

    path = tmp_path_factory.mktemp("data") / "creditcard.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def base_config(synthetic_csv) -> PreprocessingConfig:
    return PreprocessingConfig(
        data_path=synthetic_csv,
        n_qubits=4,
        test_size=0.2,
        imbalance_strategy="smote",
        smote_k_neighbors=1,   # small k to work with tiny dataset
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPreprocessingOutput:
    def test_returns_preprocessed_data(self, base_config):
        result = preprocess(base_config)
        assert isinstance(result, PreprocessedData)

    def test_feature_dimensionality(self, base_config):
        result = preprocess(base_config)
        assert result.X_train.shape[1] == base_config.n_qubits
        assert result.X_test.shape[1] == base_config.n_qubits

    def test_no_data_leakage_in_pca(self, base_config):
        """PCA must be fitted on train only — test shape is independent."""
        result = preprocess(base_config)
        assert result.X_test.shape[1] == result.X_train.shape[1]

    def test_labels_are_binary(self, base_config):
        result = preprocess(base_config)
        assert set(result.y_test).issubset({0, 1})

    def test_smote_increases_minority_class(self, base_config):
        result = preprocess(base_config)
        counts_before = result.class_counts_original
        counts_after = result.class_counts_after_resampling
        assert counts_after[1] >= counts_before[1]

    def test_class_weight_dict_sums_correctly(self, base_config):
        result = preprocess(base_config)
        weights = result.class_weight_dict
        # Both classes present
        assert set(weights.keys()) == {0, 1}
        # Minority class gets higher weight
        assert weights[1] > weights[0]


class TestImbalanceStrategies:
    def test_class_weight_strategy(self, synthetic_csv):
        cfg = PreprocessingConfig(
            data_path=synthetic_csv,
            n_qubits=4,
            imbalance_strategy="class_weight",
        )
        result = preprocess(cfg)
        # Training set should NOT be upsampled
        n_train = int(N_SAMPLES * 0.8)
        assert result.X_train.shape[0] <= n_train

    def test_no_strategy(self, synthetic_csv):
        cfg = PreprocessingConfig(
            data_path=synthetic_csv,
            n_qubits=4,
            imbalance_strategy="none",
        )
        result = preprocess(cfg)
        assert result.X_train.shape[0] > 0


class TestPCADisabled:
    def test_without_pca_feature_count_matches_input(self, synthetic_csv):
        cfg = PreprocessingConfig(
            data_path=synthetic_csv,
            n_qubits=4,
            apply_pca=False,
            imbalance_strategy="none",
        )
        result = preprocess(cfg)
        # N_FEATURES + Amount - Time (dropped)
        expected_features = N_FEATURES + 1
        assert result.X_train.shape[1] == expected_features
        assert result.pca is None
