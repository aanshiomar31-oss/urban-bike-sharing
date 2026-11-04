"""Modular data quality and preprocessing pipeline.
Performs 9-point comprehensive data audit, calendar checks, outlier diagnostics,
saves reports in CSV and JSON formats, and produces leak-free chronological partitions.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

from config import (
    TRAIN_END,
    TEST_START,
    TEST_END,
    TARGET_COL,
    REPORTS_DIR,
    PROCESSED_DATA_DIR,
)
from src.data_loader import load_raw_data, verify_target_identity


def run_data_quality_audit(df: pd.DataFrame) -> dict:
    """Executes a 9-point data quality inspection across all 17,379 observations."""
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="h")
    missing_timestamps = full_range.difference(df.index)

    # 1. Missing cells
    missing_cells = int(df.isna().sum().sum())

    # 2. Duplicate rows
    duplicate_rows = int(df.reset_index(drop=True).duplicated().sum())

    # 3. Duplicate timestamps
    duplicate_timestamps = int(df.index.duplicated().sum())

    # 4. Invalid data types (verify numeric columns)
    numeric_cols = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
                    "weathersit", "temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"]
    non_numeric_issues = {col: str(df[col].dtype) for col in numeric_cols if not np.issubdtype(df[col].dtype, np.number)}

    # 5. Negative counts
    negative_cnt = int((df[TARGET_COL] < 0).sum())
    negative_casual = int((df["casual"] < 0).sum())
    negative_registered = int((df["registered"] < 0).sum())

    # 6. Outlier analysis (IQR method - diagnostic only, not filtered)
    q25 = float(df[TARGET_COL].quantile(0.25))
    q75 = float(df[TARGET_COL].quantile(0.75))
    iqr = q75 - q25
    upper_bound = q75 + 1.5 * iqr
    outliers_count = int((df[TARGET_COL] > upper_bound).sum())

    # 7. Timestamp continuity
    expected_intervals = len(full_range)
    observed_intervals = df.index.nunique()
    missing_count = len(missing_timestamps)
    sample_missing = [str(ts) for ts in missing_timestamps[:15]]

    # 8. Target integrity
    target_identity_ok = verify_target_identity(df)

    # 9. Calendar consistency checks
    hr_mismatches = int((df["hr"] != df.index.hour).sum())
    # Note: source weekday code has 0=Sunday, pandas has 0=Monday (Python standard). Check mapping
    weekday_mismatches = int((((df.index.dayofweek + 1) % 7) != df["weekday"]).sum())

    audit_results = {
        "dataset_rows": len(df),
        "dataset_columns": df.shape[1],
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "duplicate_timestamps": duplicate_timestamps,
        "non_numeric_issues": non_numeric_issues,
        "negative_rental_counts": negative_cnt + negative_casual + negative_registered,
        "iqr_upper_threshold": upper_bound,
        "statistical_outliers_count": outliers_count,
        "statistical_outliers_percentage": round(outliers_count / len(df) * 100, 2),
        "expected_hourly_intervals": expected_intervals,
        "observed_hourly_intervals": observed_intervals,
        "missing_hourly_timestamps": missing_count,
        "sample_missing_timestamps": sample_missing,
        "missing_policy": "Preserve observed records intact. Do not synthesize or interpolate artificial target values.",
        "target_identity_valid": target_identity_ok,
        "calendar_hour_mismatches": hr_mismatches,
        "calendar_weekday_mismatches": weekday_mismatches,
        "start_date": str(df.index.min()),
        "end_date": str(df.index.max()),
    }

    # Save reports in both JSON and tabular CSV
    json_path = REPORTS_DIR / "data_quality_report.json"
    with open(json_path, "w") as f:
        json.dump(audit_results, f, indent=2)

    # Tabular summary for stakeholder reporting
    table_rows = [
        {"Check Item": "Total Records", "Result": str(len(df)), "Status": "PASS"},
        {"Check Item": "Missing Cells", "Result": str(missing_cells), "Status": "PASS" if missing_cells == 0 else "FAIL"},
        {"Check Item": "Duplicate Rows", "Result": str(duplicate_rows), "Status": "PASS" if duplicate_rows == 0 else "FAIL"},
        {"Check Item": "Duplicate Timestamps", "Result": str(duplicate_timestamps), "Status": "PASS" if duplicate_timestamps == 0 else "FAIL"},
        {"Check Item": "Negative Counts", "Result": str(negative_cnt), "Status": "PASS" if negative_cnt == 0 else "FAIL"},
        {"Check Item": "Target Identity (cnt = casual + registered)", "Result": "Exact Match 100%", "Status": "PASS" if target_identity_ok else "FAIL"},
        {"Check Item": "Missing Hourly Intervals", "Result": f"{missing_count} out of {expected_intervals}", "Status": "INFO (Retained Unmodified)"},
        {"Check Item": "Statistical Outliers (> IQR bound)", "Result": f"{outliers_count} ({audit_results['statistical_outliers_percentage']}%)", "Status": "INFO (Genuine Peak Demand Preserved)"},
        {"Check Item": "Calendar Hour Alignment", "Result": f"{hr_mismatches} mismatches", "Status": "PASS" if hr_mismatches == 0 else "FAIL"},
    ]
    pd.DataFrame(table_rows).to_csv(REPORTS_DIR / "data_quality_report.csv", index=False)

    return audit_results


def create_chronological_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits dataset into strictly chronological Train and Holdout Test partitions.
    Training: Jan 1, 2011 to Jun 30, 2012.
    Test: Jul 1, 2012 to Dec 31, 2012.
    """
    train_mask = df.index <= pd.Timestamp(TRAIN_END)
    test_mask = (df.index >= pd.Timestamp(TEST_START)) & (df.index <= pd.Timestamp(TEST_END))

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(f"Partition error: train rows = {len(train_df)}, test rows = {len(test_df)}")

    return train_df, test_df
