"""Experiment B: Autoregressive Forecasting Module.
Predicts hourly demand using calendar, temporal, weather, exact-time lags, and rolling stats.
Evaluates rolling one-hour-ahead operational prediction conditions.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor

from config import AUTOREGRESSIVE_FEATURES, TARGET_COL, RANDOM_SEED, METRICS_DIR, FORECASTS_DIR
from src.evaluation import calculate_regression_metrics
from src.direct_forecasting import compute_seasonal_naive_direct


def build_ar_models(seed: int = RANDOM_SEED) -> dict:
    """Builds candidate models for autoregressive forecasting with robust NaN imputation where needed."""
    return {
        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=100.0, random_state=seed))
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", RandomForestRegressor(
                n_estimators=250,
                max_depth=18,
                min_samples_leaf=2,
                max_features=0.85,
                n_jobs=-1,
                random_state=seed
            ))
        ]),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=40,
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


def run_autoregressive_forecasting_experiment(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Executes Experiment B across all models and returns metrics table, predictions, and fitted models."""
    X_train = train_df[AUTOREGRESSIVE_FEATURES]
    y_train = train_df[TARGET_COL]
    X_test = test_df[AUTOREGRESSIVE_FEATURES]
    y_test = test_df[TARGET_COL]

    predictions = {}
    fitted_models = {}

    # 1. Seasonal Naive baseline
    naive_pred = compute_seasonal_naive_direct(train_df, test_df)
    predictions["Seasonal Naive (168h)"] = naive_pred

    # 2. Machine learning models
    model_dict = build_ar_models()
    for name, model in model_dict.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        pred = np.clip(pred, 0, None)
        predictions[name] = pred
        fitted_models[name] = model

    # Calculate metrics
    rows = []
    for name, pred in predictions.items():
        m = calculate_regression_metrics(y_test, pred, test_df.index.hour)
        rows.append({"experiment": "B_Autoregressive", "model": name, **m})

    metrics_df = pd.DataFrame(rows).sort_values("MAE")
    metrics_df.to_csv(METRICS_DIR / "autoregressive_forecasting_metrics.csv", index=False)

    pred_df = pd.DataFrame({"actual": y_test, **predictions}, index=test_df.index)
    pred_df.to_csv(FORECASTS_DIR / "autoregressive_forecasting_predictions.csv", index_label="datetime")

    return metrics_df, pred_df, fitted_models
