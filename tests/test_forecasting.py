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


def test_strictly_two_forecasting_experiments_and_wape():
    path = METRICS_DIR / "master_model_comparison.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert "WAPE" in df.columns
    unique_exps = set(df["experiment"].unique())
    assert unique_exps == {"A_Direct", "B_Autoregressive"}, f"Unexpected experiments found: {unique_exps}"
    assert len(df) == 10  # 5 models in Exp A + 5 models in Exp B


def test_six_enterprise_sqlite_tables_exist():
    import sqlite3
    from config import SQLITE_DB_PATH
    assert SQLITE_DB_PATH.exists()
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
        required = [
            "raw_hourly_data",
            "cleaned_hourly_data",
            "monthly_analytics",
            "model_metrics",
            "forecast_results",
            "hypothesis_results"
        ]
        for req in required:
            assert req in tables, f"Missing required SQLite table: {req}"
            count = conn.execute(f"SELECT count(*) FROM {req};").fetchone()[0]
            assert count > 0, f"SQLite table {req} is empty"

