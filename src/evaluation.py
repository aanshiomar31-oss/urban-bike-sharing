"""Model evaluation and visual diagnostics module.
Calculates comprehensive metrics (MAE, RMSE, R², Peak MAE, Peak Bias, Underprediction %, Pinball Loss),
and generates Plotly and Matplotlib visual analytics.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import ALL_PEAK_HOURS, FIGURES_DIR, METRICS_DIR


def calculate_regression_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, hours_index: pd.Series | np.ndarray = None) -> dict:
    """Calculates full operational regression evaluation suite including peak-hour errors and bias."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    r2 = float(r2_score(y_true, y_pred))
    
    overall_bias = float(np.mean(y_pred - y_true))
    underpred_pct = float(np.mean(y_pred < y_true) * 100.0)
    wape = float(np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100.0) if np.sum(y_true) > 0 else 0.0

    res = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "WAPE": round(wape, 2),
        "Bias": round(overall_bias, 2),
        "Underprediction_Percentage": round(underpred_pct, 1),
    }

    if hours_index is not None:
        hours_arr = np.asarray(hours_index)
        peak_mask = np.isin(hours_arr, ALL_PEAK_HOURS)
        if np.any(peak_mask):
            peak_y_true = y_true[peak_mask]
            peak_y_pred = y_pred[peak_mask]
            res["Peak_MAE"] = round(float(mean_absolute_error(peak_y_true, peak_y_pred)), 2)
            res["Peak_RMSE"] = round(float(mean_squared_error(peak_y_true, peak_y_pred) ** 0.5), 2)
            res["Peak_Bias"] = round(float(np.mean(peak_y_pred - peak_y_true)), 2)
            res["Peak_Underprediction_Percentage"] = round(float(np.mean(peak_y_pred < peak_y_true) * 100.0), 1)

    return res


def calculate_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.90) -> dict:
    """Calculates pinball loss and empirical coverage for quantile regression."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    loss = np.maximum(quantile * residual, (quantile - 1.0) * residual)
    coverage = float(np.mean(y_true <= y_pred))
    return {
        "quantile": quantile,
        "pinball_loss": round(float(np.mean(loss)), 3),
        "empirical_coverage": round(coverage, 4),
    }


def generate_static_figures(pred_df: pd.DataFrame, metrics_df: pd.DataFrame, daily_df: pd.DataFrame):
    """Generates clean publication-quality Matplotlib figures for outputs/figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Hourly actual vs predicted line plot (First 7 days of test)
    sample_window = pred_df.iloc[:168]
    plt.figure(figsize=(12, 4.5))
    plt.plot(sample_window.index, sample_window["actual"], label="Actual Demand", color="black", linewidth=1.8)
    for col in sample_window.columns:
        if col not in ["actual", "datetime"]:
            plt.plot(sample_window.index, sample_window[col], label=col, linestyle="--", alpha=0.85)
    plt.title("One-Week Forecast vs Actual (Holdout Test: July 2012)", fontsize=12, fontweight="bold")
    plt.xlabel("Datetime", fontsize=10)
    plt.ylabel("Hourly Rentals (cnt)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_forecast_vs_actual_week.png", dpi=200)
    plt.close()

    # 2. Residual Distribution Plot
    if "best_model" in pred_df.columns:
        best_col = "best_model"
    else:
        best_col = [c for c in pred_df.columns if c not in ["actual", "datetime"]][0]
    residuals = pred_df[best_col] - pred_df["actual"]

    plt.figure(figsize=(9, 4))
    plt.hist(residuals, bins=50, color="#1f77b4", edgecolor="black", alpha=0.7)
    plt.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
    plt.axvline(residuals.mean(), color="orange", linestyle=":", linewidth=2, label=f"Mean Bias ({residuals.mean():.1f})")
    plt.title(f"Residual Distribution ({best_col})", fontsize=12, fontweight="bold")
    plt.xlabel("Residual (Predicted - Actual)", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_residual_distribution.png", dpi=200)
    plt.close()

    # 3. Model Comparison Bar Chart
    plt.figure(figsize=(10, 4.5))
    x_pos = np.arange(len(metrics_df))
    plt.bar(x_pos - 0.2, metrics_df["MAE"], width=0.4, label="Overall MAE", color="#2ca02c")
    if "Peak_MAE" in metrics_df.columns:
        plt.bar(x_pos + 0.2, metrics_df["Peak_MAE"], width=0.4, label="Peak-Hour MAE", color="#d62728")
    plt.xticks(x_pos, metrics_df["model"], rotation=15)
    plt.ylabel("Mean Absolute Error (Rentals/Hour)", fontsize=10)
    plt.title("Model Error Comparison: Overall vs Commute Peak Hours", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_model_error_comparison.png", dpi=200)
    plt.close()

    # 4. H1 Commute Share Bar Chart
    plt.figure(figsize=(8, 4))
    mean_shares = daily_df.groupby("workingday")["commute_share"].mean() * 100
    plt.bar(["Non-Working Day", "Working Day"], mean_shares, color=["#7f7f7f", "#1f77b4"], width=0.5)
    plt.ylabel("Commute Window Demand Share (%)", fontsize=10)
    plt.title("Hypothesis 1: Commuter Concentration on Working Days", fontsize=12, fontweight="bold")
    for i, v in enumerate(mean_shares):
        plt.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
    plt.ylim(0, 65)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_h1_commute_share.png", dpi=200)
    plt.close()
