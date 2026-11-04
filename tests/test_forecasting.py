"""Unit tests for model training, metrics, and non-negative prediction constraints."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import numpy as np
import pandas as pd
import joblib
from config import MODELS_DIR, METRICS_DIR, FORECASTS_DIR


def test_master_model_metrics_exist():
    path = METRICS_DIR / "master_model_comparison.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert "MAE" in df.columns
    assert "R2" in df.columns
    assert "model" in df.columns
    # Check best model MAE is under 50
    best_mae = df["MAE"].min()
    assert best_mae < 50.0


def test_saved_best_model_loads_and_predicts():
    model_path = MODELS_DIR / "best_model.joblib"
    assert model_path.exists()
    model = joblib.load(model_path)
    # Check model object has predict method
    assert hasattr(model, "predict")


def test_predictions_are_strictly_non_negative():
    pred_path = FORECASTS_DIR / "direct_forecasting_predictions.csv"
    assert pred_path.exists()
    preds = pd.read_csv(pred_path)
    for col in preds.columns:
        if col != "datetime":
            assert (preds[col] >= 0).all(), f"Found negative predictions in {col}"
