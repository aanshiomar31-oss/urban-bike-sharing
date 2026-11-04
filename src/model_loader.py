"""Model loader and inference module for Streamlit deployment.
Caches model instances using st.cache_resource and generates single-hour real-time predictions.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from config import (
    MODELS_DIR,
    DIRECT_FEATURES,
    TARGET_COL,
    MORNING_PEAK_HOURS,
    EVENING_PEAK_HOURS,
    ALL_PEAK_HOURS,
    NIGHT_HOURS,
    BUSINESS_HOURS,
)


@st.cache_resource(show_spinner="Loading trained forecasting models...")
def load_production_models():
    """Loads the champion direct forecasting model and calibrated quantile models."""
    best_model_path = MODELS_DIR / "best_model.joblib"
    quantile_path = MODELS_DIR / "quantile_models.joblib"
    metadata_path = MODELS_DIR / "model_metadata.json"

    if not best_model_path.exists():
        raise FileNotFoundError(f"Champion model not found at {best_model_path}. Please run pipeline first.")

    champion_model = joblib.load(best_model_path)
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    quantile_dict = {}
    if quantile_path.exists():
        quantile_dict = joblib.load(quantile_path)

    return {
        "champion_model": champion_model,
        "metadata": metadata,
        "quantile_models": quantile_dict
    }


def predict_single_hour(
    dt: pd.Timestamp,
    weather_sit: int = 1,
    temp_celsius: float = 20.0,
    humidity_pct: float = 60.0,
    wind_speed_kmh: float = 12.0,
    workingday_override: int = None
) -> dict:
    """Constructs the exact 20-feature input vector for a specific timestamp and predicts expected rentals."""
    models_bundle = load_production_models()
    model = models_bundle["champion_model"]
    quantiles = models_bundle.get("quantile_models", {})

    # Compute normalized weather inputs according to dataset definitions
    temp_norm = temp_celsius / 41.0
    atemp_norm = temp_celsius / 50.0  # Approx feeling temp
    hum_norm = humidity_pct / 100.0
    wind_norm = wind_speed_kmh / 67.0

    # Determine calendar flags
    hour = dt.hour
    day_of_week = dt.dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0
    is_workingday = 1 if (is_weekend == 0) else 0
    if workingday_override is not None:
        is_workingday = int(workingday_override)

    # Cyclical encodings
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    weekday_sin = np.sin(2 * np.pi * day_of_week / 7.0)
    weekday_cos = np.cos(2 * np.pi * day_of_week / 7.0)
    month_sin = np.sin(2 * np.pi * (dt.month - 1) / 12.0)
    month_cos = np.cos(2 * np.pi * (dt.month - 1) / 12.0)

    # Operational periods
    morning_peak = 1 if hour in MORNING_PEAK_HOURS else 0
    evening_peak = 1 if hour in EVENING_PEAK_HOURS else 0
    is_peak_hour = 1 if hour in ALL_PEAK_HOURS else 0
    is_off_peak = 1 if not is_peak_hour else 0
    night_period = 1 if hour in NIGHT_HOURS else 0
    business_hours = 1 if hour in BUSINESS_HOURS else 0

    # Calendar dates
    day_of_month = dt.day
    day_of_year = dt.dayofyear
    is_holiday = 0
    is_day_before_holiday = 0
    is_day_after_holiday = 0
    week_of_year = dt.isocalendar().week
    week_of_month = (dt.day - 1) // 7 + 1
    weekday_index = day_of_week
    weekend_indicator = is_weekend
    month = dt.month
    quarter = dt.quarter
    is_month_start = 1 if dt.is_month_start else 0
    is_month_end = 1 if dt.is_month_end else 0
    days_in_month = dt.days_in_month
    year = dt.year
    season = 1 if month in [12, 1, 2] else (2 if month in [3, 4, 5] else (3 if month in [6, 7, 8] else 4))
    annual_trend = (dt - pd.Timestamp("2011-01-01")).total_seconds() / (365.25 * 86400.0)

    # Interactions
    hour_x_workingday = hour * is_workingday
    hour_x_weekend = hour * is_weekend
    month_x_hour = month * hour
    season_x_hour = season * hour
    weather_x_workingday = weather_sit * is_workingday
    temp_x_weekend = temp_norm * is_weekend
    peak_hour_x_workingday = is_peak_hour * is_workingday

    # Build input dictionary
    feature_dict = {
        "hour": hour, "hour_sin": hour_sin, "hour_cos": hour_cos,
        "morning_peak": morning_peak, "evening_peak": evening_peak,
        "is_peak_hour": is_peak_hour, "is_off_peak": is_off_peak,
        "night_period": night_period, "business_hours": business_hours,
        "day_of_week": day_of_week, "day_of_month": day_of_month, "day_of_year": day_of_year,
        "is_weekend": is_weekend, "is_workingday": is_workingday, "is_holiday": is_holiday,
        "is_day_before_holiday": is_day_before_holiday, "is_day_after_holiday": is_day_after_holiday,
        "week_of_year": week_of_year, "week_of_month": week_of_month,
        "weekday_index": weekday_index, "weekend_indicator": weekend_indicator,
        "month": month, "month_sin": month_sin, "month_cos": month_cos,
        "quarter": quarter, "is_month_start": is_month_start, "is_month_end": is_month_end,
        "days_in_month": days_in_month, "year": year, "season": season, "annual_trend": annual_trend,
        "hour_x_workingday": hour_x_workingday, "hour_x_weekend": hour_x_weekend,
        "month_x_hour": month_x_hour, "season_x_hour": season_x_hour,
        "weather_x_workingday": weather_x_workingday, "temp_x_weekend": temp_x_weekend,
        "peak_hour_x_workingday": peak_hour_x_workingday,
        "weathersit": weather_sit, "temp": temp_norm, "atemp": atemp_norm,
        "hum": hum_norm, "windspeed": wind_norm
    }

    # Construct single-row DataFrame aligned to DIRECT_FEATURES
    row_df = pd.DataFrame([feature_dict])[DIRECT_FEATURES]

    # Predict expected point demand
    pred_point = float(np.clip(model.predict(row_df)[0], 0, None))

    # Quantile intervals if available
    p50_val = pred_point
    p90_val = pred_point * 1.35  # Fallback empirical 35% peak buffer
    if quantiles and "p50_model" in quantiles and "p90_model" in quantiles:
        p50_val = float(np.clip(quantiles["p50_model"].predict(row_df)[0], 0, None))
        p90_raw = float(np.clip(quantiles["p90_model"].predict(row_df)[0], 0, None))
        offset = quantiles.get("calibration_offset", 31.4)
        p90_val = p90_raw + offset

    return {
        "timestamp": str(dt),
        "predicted_demand_point": round(pred_point, 1),
        "predicted_demand_p50": round(p50_val, 1),
        "predicted_demand_p90_buffer": round(p90_val, 1),
        "is_peak_hour": bool(is_peak_hour),
        "is_workingday": bool(is_workingday),
        "weather_situation_code": weather_sit,
        "assumptions": "Direct one-hour forecast assuming numerical weather forecast inputs are accurate."
    }
