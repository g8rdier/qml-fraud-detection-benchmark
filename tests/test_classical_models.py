"""Unit tests for src/classical_models.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.classical_models import (
    build_random_forest,
    build_xgboost,
    train_classical_model,
)

RNG = np.random.default_rng(7)
X_TRAIN = RNG.standard_normal((200, 8))
Y_TRAIN = RNG.integers(0, 2, size=200)
X_TEST  = RNG.standard_normal((50, 8))
Y_TEST  = RNG.integers(0, 2, size=50)


class TestBuilders:
    def test_rf_returns_estimator(self):
        clf = build_random_forest(n_estimators=10)
        assert hasattr(clf, "fit")

    def test_xgb_returns_estimator(self):
        clf = build_xgboost(n_estimators=10)
        assert hasattr(clf, "fit")

    def test_xgb_scale_pos_weight(self):
        clf = build_xgboost(scale_pos_weight=5.0)
        assert clf.scale_pos_weight == 5.0


class TestTrainClassicalModel:
    @pytest.fixture(scope="class")
    def rf_result(self):
        clf = build_random_forest(n_estimators=10)
        return train_classical_model(
            model=clf,
            X_train=X_TRAIN, y_train=Y_TRAIN,
            X_test=X_TEST,   y_test=Y_TEST,
            cv_folds=2,
        )

    def test_returns_y_pred(self, rf_result):
        assert "y_pred" in rf_result
        assert rf_result["y_pred"].shape == (50,)

    def test_returns_y_prob(self, rf_result):
        assert "y_prob" in rf_result
        assert rf_result["y_prob"].shape == (50, 2)

    def test_proba_sums_to_one(self, rf_result):
        np.testing.assert_allclose(
            rf_result["y_prob"].sum(axis=1), 1.0, atol=1e-6
        )

    def test_fit_time_positive(self, rf_result):
        assert rf_result["fit_time"] > 0

    def test_cv_scores_present(self, rf_result):
        assert "f1" in rf_result["cv_scores"]
        assert len(rf_result["cv_scores"]["f1"]) == 2   # cv_folds=2

    def test_model_saved(self, rf_result, tmp_path):
        import joblib
        clf = build_random_forest(n_estimators=10)
        train_classical_model(
            model=clf,
            X_train=X_TRAIN, y_train=Y_TRAIN,
            X_test=X_TEST,   y_test=Y_TEST,
            cv_folds=0,
            save_dir=tmp_path,
            model_name="test_rf",
        )
        assert (tmp_path / "test_rf.joblib").exists()
        loaded = joblib.load(tmp_path / "test_rf.joblib")
        assert hasattr(loaded, "predict")

    def test_no_cv(self):
        clf = build_xgboost(n_estimators=10)
        result = train_classical_model(
            model=clf,
            X_train=X_TRAIN, y_train=Y_TRAIN,
            X_test=X_TEST,   y_test=Y_TEST,
            cv_folds=0,
        )
        assert result["cv_scores"] == {}
