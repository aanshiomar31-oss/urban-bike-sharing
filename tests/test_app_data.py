"""Unit tests for Streamlit application data assets and model loading."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import pandas as pd
from config import METRICS_DIR, FORECASTS_DIR, MONTHLY_REPORTS_DIR, MODELS_DIR, REPORTS_DIR
from src.model_loader import load_production_models, predict_single_hour


def test_required_cloud_artifacts_exist():
    required_files = [
        MODELS_DIR / "best_model.joblib",
        MODELS_DIR / "quantile_models.joblib",
        METRICS_DIR / "direct_forecasting_metrics.csv",
        METRICS_DIR / "master_model_comparison.csv",
        METRICS_DIR / "pareto_analysis.json",
        METRICS_DIR / "hypothesis_results.json",
        METRICS_DIR / "fleet_summary.json",
        FORECASTS_DIR / "direct_forecasting_predictions.csv",
        MONTHLY_REPORTS_DIR / "monthly_summary.csv",
        REPORTS_DIR / "data_quality_report.json",
    ]
    for p in required_files:
        assert p.exists(), f"Deployment artifact missing: {p}"


def test_model_loading_and_single_hour_inference():
    bundle = load_production_models()
    assert "champion_model" in bundle
    assert hasattr(bundle["champion_model"], "predict")

    # Run single-hour test prediction
    res = predict_single_hour(
        dt=pd.Timestamp("2026-06-15 17:00:00"),
        weather_sit=1,
        temp_celsius=25.0,
        humidity_pct=50.0,
        wind_speed_kmh=12.0
    )
    assert res["predicted_demand_point"] > 0
    assert res["predicted_demand_p90_buffer"] >= res["predicted_demand_point"]
    assert res["is_peak_hour"] is True
