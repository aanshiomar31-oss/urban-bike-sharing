"""Experiment C: Quantile Forecasting Module for Operational Risk Management.
Implements P50 (central demand) and P90 (conservative safety buffer) quantile models
with Pinball Loss, empirical coverage tracking, and conformal calibration.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib

from config import DIRECT_FEATURES, TARGET_COL, RANDOM_SEED, METRICS_DIR, FORECASTS_DIR, MODELS_DIR
from src.evaluation import calculate_pinball_loss, calculate_regression_metrics


def run_quantile_forecasting_experiment(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Trains P50 and P90 quantile models, computes pinball loss, coverage, and calibration offsets."""
    X_train = train_df[DIRECT_FEATURES]
    y_train = train_df[TARGET_COL]
    X_test = test_df[DIRECT_FEATURES]
    y_test = test_df[TARGET_COL]

    # Train P50 model (Median demand estimate)
    model_p50 = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.50,
        max_iter=250,
        learning_rate=0.06,
        max_leaf_nodes=35,
        random_state=RANDOM_SEED
    )
    model_p50.fit(X_train, y_train)
    pred_p50 = np.clip(model_p50.predict(X_test), 0, None)

    # Train P90 model (Conservative upper demand planning buffer)
    model_p90 = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.90,
        max_iter=250,
        learning_rate=0.06,
        max_leaf_nodes=35,
        random_state=RANDOM_SEED
    )
    model_p90.fit(X_train, y_train)
    pred_p90_raw = np.clip(model_p90.predict(X_test), 0, None)

    # Calculate Pinball loss and coverage
    metrics_p50 = calculate_pinball_loss(y_test, pred_p50, quantile=0.50)
    metrics_p90_raw = calculate_pinball_loss(y_test, pred_p90_raw, quantile=0.90)

    # Calibrate P90 with residual offset on first half of test (calibration split)
    cal_len = len(test_df) // 3
    y_cal = y_test.iloc[:cal_len].to_numpy()
    p_cal = pred_p90_raw[:cal_len]
    cal_residuals = y_cal - p_cal
    # 90th percentile of residuals gives conformal offset
    offset = float(np.percentile(np.maximum(0, cal_residuals), 90))
    pred_p90_calibrated = pred_p90_raw + offset
    metrics_p90_cal = calculate_pinball_loss(y_test, pred_p90_calibrated, quantile=0.90)

    # Peak hour metrics
    peak_mask = test_df.index.hour.isin([7, 8, 9, 17, 18, 19])
    peak_y = y_test[peak_mask].to_numpy()
    peak_cov_raw = float(np.mean(peak_y <= pred_p90_raw[peak_mask]))
    peak_cov_cal = float(np.mean(peak_y <= pred_p90_calibrated[peak_mask]))

    summary_rows = [
        {
            "model": "HistGB_P50",
            "target_quantile": 0.50,
            "pinball_loss": metrics_p50["pinball_loss"],
            "empirical_coverage": metrics_p50["empirical_coverage"],
            "peak_hour_coverage": float(np.mean(peak_y <= pred_p50[peak_mask])),
            "mae": round(float(np.mean(np.abs(y_test - pred_p50))), 2),
            "calibration_offset": 0.0
        },
        {
            "model": "HistGB_P90_Raw",
            "target_quantile": 0.90,
            "pinball_loss": metrics_p90_raw["pinball_loss"],
            "empirical_coverage": metrics_p90_raw["empirical_coverage"],
            "peak_hour_coverage": round(peak_cov_raw, 4),
            "mae": round(float(np.mean(np.abs(y_test - pred_p90_raw))), 2),
            "calibration_offset": 0.0
        },
        {
            "model": "HistGB_P90_Calibrated",
            "target_quantile": 0.90,
            "pinball_loss": metrics_p90_cal["pinball_loss"],
            "empirical_coverage": metrics_p90_cal["empirical_coverage"],
            "peak_hour_coverage": round(peak_cov_cal, 4),
            "mae": round(float(np.mean(np.abs(y_test - pred_p90_calibrated))), 2),
            "calibration_offset": round(offset, 2)
        }
    ]

    metrics_df = pd.DataFrame(summary_rows)
    metrics_df.to_csv(METRICS_DIR / "quantile_forecasting_metrics.csv", index=False)

    pred_df = pd.DataFrame({
        "actual": y_test,
        "p50_median": pred_p50,
        "p90_conservative_raw": pred_p90_raw,
        "p90_conservative_calibrated": pred_p90_calibrated
    }, index=test_df.index)
    pred_df.to_csv(FORECASTS_DIR / "quantile_forecasting_predictions.csv", index_label="datetime")

    fitted_models = {
        "p50_model": model_p50,
        "p90_model": model_p90,
        "calibration_offset": offset
    }
    joblib.dump(fitted_models, MODELS_DIR / "quantile_models.joblib")

    return metrics_df, pred_df, fitted_models
