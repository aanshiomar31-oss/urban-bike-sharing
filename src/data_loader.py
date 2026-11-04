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
