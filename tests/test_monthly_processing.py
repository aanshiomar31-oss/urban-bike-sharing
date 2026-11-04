"""Unit tests for monthly processing engine."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import pandas as pd
from config import MONTHLY_REPORTS_DIR


def test_monthly_summary_generated_with_24_months():
    summary_path = MONTHLY_REPORTS_DIR / "monthly_summary.csv"
    assert summary_path.exists(), "monthly_summary.csv was not generated"
    df = pd.read_csv(summary_path)
    assert len(df) == 24, f"Expected 24 months, found {len(df)}"
    assert "year_month" in df.columns
    assert "total_rentals" in df.columns
    assert "Holdout_test_mae" in [c.capitalize() for c in df.columns] or "holdout_test_mae" in df.columns


def test_individual_monthly_dossiers_exist():
    for ym in ["2011-01", "2011-07", "2012-01", "2012-12"]:
        dossier = MONTHLY_REPORTS_DIR / f"{ym}.json"
        assert dossier.exists(), f"Monthly dossier for {ym} does not exist"
