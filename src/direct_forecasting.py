"""Experiment A: Direct Forecasting Module.
Predicts hourly rental demand directly from calendar, temporal, and weather signals
WITHOUT target-derived lag or rolling features.
Models: Seasonal Naive, Ridge Regression, Random Forest, HistGradientBoosting, XGBoost.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor

from config import DIRECT_FEATURES, TARGET_COL, RANDOM_SEED, METRICS_DIR, FORECASTS_DIR
from src.evaluation import calculate_regression_metrics


def build_direct_models(seed: int = RANDOM_SEED) -> dict:
    """Instantiates the five candidate forecasting models with reproducible configurations."""
    return {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=100.0, random_state=seed))
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=250,
            max_depth=16,
            min_samples_leaf=2,
            max_features=0.85,
            n_jobs=-1,
            random_state=seed
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=35,
            min_samples_leaf=15,
            l2_regularization=1.5,
            random_state=seed
        ),
        "XGBoost": XGBRegressor(
            n_estimators=400,
            learning_rate=0.04,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=3.0,
            random_state=seed,
            n_jobs=-1
        )
    }


def compute_seasonal_naive_direct(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Seasonal Naive baseline using exact 168-hour (1 week prior) lookup with 24h fallback."""
    full_history = pd.concat([train_df[TARGET_COL], test_df[TARGET_COL]])
    target_idx = test_df.index
    
    # 168h prior
    base_idx = target_idx - pd.Timedelta(hours=168)
    base_pred = full_history.reindex(base_idx).to_numpy()
    
    # 24h fallback
    fallback_24h = full_history.reindex(target_idx - pd.Timedelta(hours=24)).to_numpy()
    base_pred = np.where(np.isfinite(base_pred), base_pred, fallback_24h)
    
    # Training median fallback
    train_median = float(train_df[TARGET_COL].median())
    base_pred = np.where(np.isfinite(base_pred), base_pred, train_median)
    return base_pred


def run_direct_forecasting_experiment(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Executes Experiment A across all 5 models and returns metrics table, predictions, and fitted models."""
    X_train = train_df[DIRECT_FEATURES]
    y_train = train_df[TARGET_COL]
    X_test = test_df[DIRECT_FEATURES]
    y_test = test_df[TARGET_COL]

    predictions = {}
    fitted_models = {}

    # 1. Seasonal Naive baseline
    naive_pred = compute_seasonal_naive_direct(train_df, test_df)
    predictions["Seasonal Naive (168h)"] = naive_pred

    # 2. Machine learning models
    model_dict = build_direct_models()
    for name, model in model_dict.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        # Demand cannot be negative in physical operations
        pred = np.clip(pred, 0, None)
        predictions[name] = pred
        fitted_models[name] = model

    # Calculate metrics
    rows = []
    for name, pred in predictions.items():
        m = calculate_regression_metrics(y_test, pred, test_df.index.hour)
        rows.append({"experiment": "A_Direct", "model": name, **m})

    metrics_df = pd.DataFrame(rows).sort_values("MAE")
    metrics_df.to_csv(METRICS_DIR / "direct_forecasting_metrics.csv", index=False)

    pred_df = pd.DataFrame({"actual": y_test, **predictions}, index=test_df.index)
    pred_df.to_csv(FORECASTS_DIR / "direct_forecasting_predictions.csv", index_label="datetime")

    return metrics_df, pred_df, fitted_models
