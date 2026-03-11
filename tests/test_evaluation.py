"""Unit tests for src/evaluation.py."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.evaluation import ModelMetrics, evaluate_model, save_metrics_json


Y_TRUE = np.array([0, 0, 0, 1, 1, 1, 0, 1])
Y_PRED = np.array([0, 0, 1, 1, 1, 0, 0, 1])
Y_PROB = np.array([0.1, 0.2, 0.8, 0.9, 0.85, 0.4, 0.15, 0.95])


def test_evaluate_returns_metrics():
    m = evaluate_model("TestModel", Y_TRUE, Y_PRED, Y_PROB)
    assert isinstance(m, ModelMetrics)


def test_f1_fraud_between_zero_and_one():
    m = evaluate_model("TestModel", Y_TRUE, Y_PRED, Y_PROB)
    assert 0.0 <= m.f1_fraud <= 1.0


def test_mcc_range():
    m = evaluate_model("TestModel", Y_TRUE, Y_PRED, Y_PROB)
    assert -1.0 <= m.mcc <= 1.0


def test_pr_auc_available_with_proba():
    m = evaluate_model("TestModel", Y_TRUE, Y_PRED, Y_PROB)
    assert not math.isnan(m.pr_auc)


def test_pr_auc_nan_without_proba():
    m = evaluate_model("TestModel", Y_TRUE, Y_PRED, y_prob=None)
    assert math.isnan(m.pr_auc)


def test_save_metrics_json(tmp_path):
    m = evaluate_model("TestModel", Y_TRUE, Y_PRED, Y_PROB)
    out = tmp_path / "metrics.json"
    save_metrics_json([m], out)
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0]["model_name"] == "TestModel"
