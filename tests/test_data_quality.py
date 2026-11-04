"""Unit tests for dataset ingestion, target identity, and 9-point data audit."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import pandas as pd
from src.data_loader import load_raw_data, verify_target_identity
from src.preprocessing import run_data_quality_audit
from config import DIRECT_FEATURES, AUTOREGRESSIVE_FEATURES, EXCLUDED_COLS


def test_data_ingestion_and_target_identity():
    df = load_raw_data()
    assert len(df) == 17379
    assert df.index.name == "datetime"
    assert verify_target_identity(df) is True


def test_audit_flags_missing_timestamps_without_synthesizing():
    df = load_raw_data()
    audit = run_data_quality_audit(df)
    assert audit["missing_cells"] == 0
    assert audit["duplicate_rows"] == 0
    assert audit["duplicate_timestamps"] == 0
    assert audit["negative_rental_counts"] == 0
    assert audit["missing_hourly_timestamps"] == 165
    assert audit["expected_hourly_intervals"] == 17544
    assert audit["observed_hourly_intervals"] == 17379


def test_feature_lists_have_zero_target_leakage():
    for col in ["cnt", "casual", "registered", "instant"]:
        assert col not in DIRECT_FEATURES
        assert col not in AUTOREGRESSIVE_FEATURES
