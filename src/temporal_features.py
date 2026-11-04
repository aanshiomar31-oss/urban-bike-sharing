"""Multi-resolution temporal feature engineering module.
Constructs features across the hierarchy: Year -> Quarter -> Month -> Week -> Day -> Hour,
with mathematically rigorous cyclical sine/cosine encodings and operational interactions.
"""

import numpy as np
import pandas as pd
from config import (
    MORNING_PEAK_HOURS,
    EVENING_PEAK_HOURS,
    ALL_PEAK_HOURS,
    NIGHT_HOURS,
    BUSINESS_HOURS,
)


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw hourly timestamps into a multi-resolution temporal feature matrix.
    
    Cyclical Encoding Principle:
    Standard ordinal time indices (e.g. hour 0..23) impose an artificial numerical cliff
    where 23:00 and 00:00 appear 23 units apart. Mapping time onto a unit circle via:
        sin(2 * pi * t / T),  cos(2 * pi * t / T)
    ensures Euclidean distance in feature space matches true chronological elapsed distance.
    """
    out = df.copy()
    idx = out.index

    # ----------------------------------------------------
    # 1. Hourly Features
    # ----------------------------------------------------
    out["hour"] = idx.hour
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24.0)
    out["morning_peak"] = idx.hour.isin(MORNING_PEAK_HOURS).astype(int)
    out["evening_peak"] = idx.hour.isin(EVENING_PEAK_HOURS).astype(int)
    out["is_peak_hour"] = idx.hour.isin(ALL_PEAK_HOURS).astype(int)
    out["is_off_peak"] = (~idx.hour.isin(ALL_PEAK_HOURS)).astype(int)
    out["night_period"] = idx.hour.isin(NIGHT_HOURS).astype(int)
    out["business_hours"] = idx.hour.isin(BUSINESS_HOURS).astype(int)

    # ----------------------------------------------------
    # 2. Daily Features
    # ----------------------------------------------------
    out["day_of_week"] = idx.dayofweek
    out["weekday_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7.0)
    out["weekday_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7.0)
    out["day_of_month"] = idx.day
    out["day_of_year"] = idx.dayofyear
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["is_workingday"] = out["workingday"].astype(int)
    out["is_holiday"] = out["holiday"].astype(int)

    # Pre/Post Holiday Flags (calculated at calendar day level)
    daily_holidays = out.groupby(out.index.normalize())["is_holiday"].max()
    is_before_hol = daily_holidays.shift(-1).fillna(0).reindex(out.index.normalize()).to_numpy()
    is_after_hol = daily_holidays.shift(1).fillna(0).reindex(out.index.normalize()).to_numpy()
    out["is_day_before_holiday"] = is_before_hol.astype(int)
    out["is_day_after_holiday"] = is_after_hol.astype(int)

    # ----------------------------------------------------
    # 3. Weekly Features
    # ----------------------------------------------------
    out["week_of_year"] = idx.isocalendar().week.astype(int).to_numpy()
    out["week_of_month"] = ((idx.day - 1) // 7 + 1).astype(int)
    out["weekday_index"] = idx.dayofweek
    out["weekend_indicator"] = out["is_weekend"]

    # ----------------------------------------------------
    # 4. Monthly Features
    # ----------------------------------------------------
    out["month"] = idx.month
    out["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12.0)
    out["quarter"] = idx.quarter
    out["is_month_start"] = idx.is_month_start.astype(int)
    out["is_month_end"] = idx.is_month_end.astype(int)
    out["days_in_month"] = idx.days_in_month

    # ----------------------------------------------------
    # 5. Annual and Seasonal Features
    # ----------------------------------------------------
    out["year"] = idx.year
    # annual_trend is elapsed time in fractional years from dataset start
    dataset_start = idx.min()
    out["annual_trend"] = (idx - dataset_start).total_seconds() / (365.25 * 86400.0)

    # ----------------------------------------------------
    # 6. Interaction Features (Domain-grounded)
    # ----------------------------------------------------
    out["hour_x_workingday"] = out["hour"] * out["is_workingday"]
    out["hour_x_weekend"] = out["hour"] * out["is_weekend"]
    out["month_x_hour"] = out["month"] * out["hour"]
    out["season_x_hour"] = out["season"] * out["hour"]
    out["weather_x_workingday"] = out["weathersit"] * out["is_workingday"]
    out["temp_x_weekend"] = out["temp"] * out["is_weekend"]
    out["peak_hour_x_workingday"] = out["is_peak_hour"] * out["is_workingday"]

    return out
