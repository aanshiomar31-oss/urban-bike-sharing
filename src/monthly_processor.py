"""Month-wise hourly processing engine module.
Generates individual monthly intelligence dossiers (24 months across 2011-2012),
computes rider composition, volatility, busiest periods, and exports a combined summary.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from config import (
    TARGET_COL,
    CASUAL_COL,
    REGISTERED_COL,
    ALL_PEAK_HOURS,
    MONTHLY_REPORTS_DIR,
    METRICS_DIR,
)


def process_all_months(df: pd.DataFrame, test_pred_df: pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Processes each year-month (24 calendar months) into a rich operational summary table and individual JSON reports."""
    MONTHLY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_copy = df.copy()
    df_copy["year_month"] = df_copy.index.to_period("M").astype(str)
    
    unique_ym = sorted(df_copy["year_month"].unique())
    monthly_summary_records = []

    for ym in unique_ym:
        m_df = df_copy[df_copy["year_month"] == ym]
        
        # 1. Continuity check
        full_m_range = pd.date_range(m_df.index.min(), m_df.index.max(), freq="h")
        missing_count = len(full_m_range.difference(m_df.index))
        
        # 2. Key totals and central tendencies
        total_rentals = int(m_df[TARGET_COL].sum())
        mean_hourly = float(m_df[TARGET_COL].mean())
        median_hourly = float(m_df[TARGET_COL].median())
        peak_hourly = int(m_df[TARGET_COL].max())
        peak_timestamp = str(m_df[TARGET_COL].idxmax())
        
        # 3. Busiest day
        daily_sum = m_df.groupby(m_df.index.normalize())[TARGET_COL].sum()
        busiest_day = str(daily_sum.idxmax().date())
        busiest_day_demand = int(daily_sum.max())
        
        # 4. Weekday vs Weekend
        weekday_demand = float(m_df[m_df["workingday"] == 1][TARGET_COL].mean()) if (m_df["workingday"] == 1).any() else 0.0
        weekend_demand = float(m_df[m_df["is_weekend"] == 1][TARGET_COL].mean()) if (m_df["is_weekend"] == 1).any() else 0.0
        
        # 5. Peak vs Off-peak
        peak_demand = float(m_df[m_df.index.hour.isin(ALL_PEAK_HOURS)][TARGET_COL].mean())
        off_peak_demand = float(m_df[~m_df.index.hour.isin(ALL_PEAK_HOURS)][TARGET_COL].mean())
        
        # 6. Volatility (Coefficient of variation: std / mean)
        std_hourly = float(m_df[TARGET_COL].std())
        volatility = round(std_hourly / (mean_hourly + 1e-6), 3)
        
        # 7. Rider composition
        casual_total = int(m_df[CASUAL_COL].sum())
        registered_total = int(m_df[REGISTERED_COL].sum())
        casual_pct = round(casual_total / total_rentals * 100, 2) if total_rentals > 0 else 0.0
        registered_pct = round(registered_total / total_rentals * 100, 2) if total_rentals > 0 else 0.0
        
        # 8. Weather stats
        avg_temp_c = round(float(m_df["temp"].mean() * 41.0), 1)
        clear_demand = float(m_df[m_df["weathersit"] == 1][TARGET_COL].mean()) if (m_df["weathersit"] == 1).any() else 0.0
        adverse_demand = float(m_df[m_df["weathersit"] >= 3][TARGET_COL].mean()) if (m_df["weathersit"] >= 3).any() else 0.0

        # 9. Monthly accuracy (if test predictions cover this month)
        monthly_mae = None
        if test_pred_df is not None:
            pred_m = test_pred_df[test_pred_df.index.to_period("M").astype(str) == ym]
            if len(pred_m) > 50:
                best_col = [c for c in pred_m.columns if c not in ["actual", "datetime"]][0]
                monthly_mae = round(float(np.mean(np.abs(pred_m["actual"] - pred_m[best_col]))), 2)

        monthly_dict = {
            "year_month": ym,
            "observations_count": len(m_df),
            "missing_hourly_intervals": missing_count,
            "total_rentals": total_rentals,
            "mean_hourly_demand": round(mean_hourly, 1),
            "median_hourly_demand": round(median_hourly, 1),
            "peak_hourly_demand": peak_hourly,
            "peak_hourly_timestamp": peak_timestamp,
            "busiest_day": busiest_day,
            "busiest_day_demand": busiest_day_demand,
            "weekday_avg_hourly": round(weekday_demand, 1),
            "weekend_avg_hourly": round(weekend_demand, 1),
            "commute_peak_avg_hourly": round(peak_demand, 1),
            "off_peak_avg_hourly": round(off_peak_demand, 1),
            "demand_volatility_cov": volatility,
            "casual_rentals": casual_total,
            "registered_rentals": registered_total,
            "casual_percentage": casual_pct,
            "registered_percentage": registered_pct,
            "average_temperature_celsius": avg_temp_c,
            "clear_weather_avg_demand": round(clear_demand, 1),
            "adverse_weather_avg_demand": round(adverse_demand, 1),
            "holdout_test_mae": monthly_mae
        }

        # Save individual monthly dossier JSON
        with open(MONTHLY_REPORTS_DIR / f"{ym}.json", "w") as f:
            json.dump(monthly_dict, f, indent=2)

        monthly_summary_records.append(monthly_dict)

    summary_df = pd.DataFrame(monthly_summary_records)
    summary_df.to_csv(MONTHLY_REPORTS_DIR / "monthly_summary.csv", index=False)
    summary_df.to_csv(METRICS_DIR / "monthly_summary.csv", index=False)

    # Month-Hour Heatmap matrix (24 months x 24 hours)
    heatmap_matrix = df_copy.pivot_table(index="year_month", columns="hour", values=TARGET_COL, aggfunc="mean").round(1)
    heatmap_matrix.to_csv(MONTHLY_REPORTS_DIR / "month_hour_heatmap_matrix.csv")

    return summary_df, heatmap_matrix
