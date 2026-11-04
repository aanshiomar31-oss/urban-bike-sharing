"""Data loader and database initialization module.
Loads UCI Capital Bikeshare hour.csv, validates columns, parses continuous timestamps,
verifies target identity (cnt = casual + registered), and provides SQLite persistence.
"""

from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
from config import RAW_DATA_PATH, SQLITE_DB_PATH, TARGET_COL, CASUAL_COL, REGISTERED_COL

REQUIRED_COLUMNS = {
    "instant", "dteday", "season", "yr", "mnth", "hr",
    "holiday", "weekday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"
}

DATA_DICTIONARY = {
    "instant": "Record index (integer 1..17379)",
    "dteday": "Date string (YYYY-MM-DD)",
    "season": "Season code (1: winter, 2: spring, 3: summer, 4: fall)",
    "yr": "Year (0: 2011, 1: 2012)",
    "mnth": "Month of the year (1 to 12)",
    "hr": "Hour of the day (0 to 23)",
    "holiday": "Whether day is holiday (0: no, 1: yes)",
    "weekday": "Day of the week (0: Sunday, 1: Monday, ... 6: Saturday)",
    "workingday": "Working day indicator (1: neither weekend nor holiday, 0: otherwise)",
    "weathersit": "Weather code (1: Clear/Few clouds, 2: Mist/Cloudy, 3: Light Rain/Snow, 4: Heavy Rain/Ice/Fog)",
    "temp": "Normalized temperature in Celsius (divided by 41)",
    "atemp": "Normalized feeling temperature in Celsius (divided by 50)",
    "hum": "Normalized relative humidity (divided by 100)",
    "windspeed": "Normalized wind speed (divided by 67)",
    "casual": "Count of rentals by unregistered/casual riders",
    "registered": "Count of rentals by registered/membership riders",
    "cnt": "Total count of rental bikes (target: cnt = casual + registered)",
}


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Loads hour.csv, verifies columns, parses timestamps, and sorts chronologically."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Parse exact hourly timestamp
    df["datetime"] = pd.to_datetime(df["dteday"], errors="coerce") + pd.to_timedelta(df["hr"], unit="h")
    if df["datetime"].isna().any():
        raise ValueError("Failed to parse valid timestamps for one or more records.")

    # Sort strictly chronologically and set index
    df = df.sort_values("datetime").reset_index(drop=True)
    df.index = pd.DatetimeIndex(df["datetime"])
    return df


def verify_target_identity(df: pd.DataFrame) -> bool:
    """Verifies that cnt = casual + registered across 100% of rows."""
    matches = (df[TARGET_COL] == df[CASUAL_COL] + df[REGISTERED_COL]).all()
    if not matches:
        mismatch_count = (df[TARGET_COL] != df[CASUAL_COL] + df[REGISTERED_COL]).sum()
        raise ValueError(f"Target identity violated in {mismatch_count} rows!")
    return bool(matches)


def get_dataset_profile(df: pd.DataFrame) -> dict:
    """Generates a comprehensive descriptive dictionary of the ingested dataset."""
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="h")
    missing_timestamps = full_range.difference(df.index)

    return {
        "dataset_source": "UCI Machine Learning Repository - Capital Bikeshare (Washington, D.C.)",
        "file_name": RAW_DATA_PATH.name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "start_timestamp": str(df.index.min()),
        "end_timestamp": str(df.index.max()),
        "expected_hourly_intervals": int(len(full_range)),
        "observed_hourly_intervals": int(df.index.nunique()),
        "missing_timestamps_count": int(len(missing_timestamps)),
        "target_identity_valid": verify_target_identity(df),
        "total_rentals": int(df[TARGET_COL].sum()),
        "total_casual_rentals": int(df[CASUAL_COL].sum()),
        "total_registered_rentals": int(df[REGISTERED_COL].sum()),
        "mean_hourly_demand": float(df[TARGET_COL].mean()),
        "median_hourly_demand": float(df[TARGET_COL].median()),
        "max_hourly_demand": int(df[TARGET_COL].max()),
        "min_hourly_demand": int(df[TARGET_COL].min()),
        "data_dictionary": DATA_DICTIONARY,
    }


def save_to_sqlite(df: pd.DataFrame, table_name: str = "raw_hourly", db_path: Path = SQLITE_DB_PATH):
    """Persists a DataFrame to SQLite for lightweight relational querying without external dependencies."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        export_df = df.copy()
        if "datetime" in export_df.columns:
            export_df["datetime"] = export_df["datetime"].astype(str)
        else:
            export_df["datetime"] = export_df.index.astype(str)
        export_df.to_sql(table_name, conn, if_exists="replace", index=False)


def populate_enterprise_sqlite_tables(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    hypothesis_dict: dict,
    db_path: Path = SQLITE_DB_PATH
):
    """Populates the 6 enterprise database tables required by the 5-layer system architecture:
    1. raw_hourly_data
    2. cleaned_hourly_data
    3. monthly_analytics
    4. model_metrics
    5. forecast_results
    6. hypothesis_results
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Format hypothesis dictionary as DataFrame
    hyp_rows = []
    for k, v in hypothesis_dict.items():
        if isinstance(v, dict):
            hyp_rows.append({
                "test_key": k,
                "hypothesis": v.get("hypothesis", ""),
                "unit_of_analysis": v.get("unit_of_analysis", ""),
                "working_days_sample": v.get("working_days_sample_size", v.get("hot_days_count", None)),
                "nonworking_days_sample": v.get("nonworking_days_sample_size", v.get("milder_days_count", None)),
                "working_days_mean": v.get("working_days_mean_share", v.get("hot_days_median_demand", None)),
                "nonworking_days_mean": v.get("nonworking_days_mean_share", v.get("milder_days_median_demand", None)),
                "observed_difference": v.get("observed_difference_percentage_points", None),
                "test_statistic": v.get("test_statistic", v.get("mann_whitney_u_statistic", None)),
                "p_value": v.get("one_sided_p_value", v.get("two_sided_p_value", None)),
                "bootstrap_ci_95": str(v.get("bootstrap_95_ci_diff_percentage_points", "")),
                "reject_null": int(v.get("reject_null_hypothesis", v.get("reject_null", False))),
                "statistical_interpretation": v.get("statistical_interpretation", v.get("business_note", "")),
                "business_interpretation": v.get("business_interpretation", "")
            })
    hyp_df = pd.DataFrame(hyp_rows)

    targets = [db_path]
    root_db = db_path.parent.parent / "bikeshare.db"
    if root_db != db_path:
        targets.append(root_db)

    for path in targets:
        with sqlite3.connect(path) as conn:
            # 1. raw_hourly_data
            save_to_sqlite(raw_df, "raw_hourly_data", path)
            save_to_sqlite(raw_df, "raw_hourly", path)  # backwards compatibility

            # 2. cleaned_hourly_data
            save_to_sqlite(cleaned_df, "cleaned_hourly_data", path)
            save_to_sqlite(cleaned_df, "featured_hourly", path)  # backwards compatibility

            # 3. monthly_analytics
            monthly_export = monthly_df.copy()
            monthly_export.to_sql("monthly_analytics", conn, if_exists="replace", index=False)

            # 4. model_metrics
            metrics_export = metrics_df.copy()
            metrics_export.to_sql("model_metrics", conn, if_exists="replace", index=False)

            # 5. forecast_results
            forecast_export = forecast_df.copy()
            if "datetime" not in forecast_export.columns:
                forecast_export["datetime"] = forecast_export.index.astype(str)
            else:
                forecast_export["datetime"] = forecast_export["datetime"].astype(str)
            forecast_export.to_sql("forecast_results", conn, if_exists="replace", index=False)

            # 6. hypothesis_results
            hyp_df.to_sql("hypothesis_results", conn, if_exists="replace", index=False)

