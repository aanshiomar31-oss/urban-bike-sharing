"""Unit tests for historical lag and rolling statistics, verifying no future target leakage."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import numpy as np
import pandas as pd
from src.data_loader import load_raw_data
from src.lag_features import create_lag_features


def test_lag_1_exact_previous_observation():
    df = load_raw_data()
    out = create_lag_features(df)
    
    # At 2011-01-01 02:00, prior hour 01:00 actual cnt was 40. lag_1 should be 40.
    t_check = pd.Timestamp("2011-01-01 02:00:00")
    if t_check in out.index:
        assert out.loc[t_check, "lag_1"] == 40.0

    # At 2011-01-02 06:00, the 05:00 timestamp is missing. Exact lag_1 must be NaN, NOT 04:00 value.
    missing_pred_t = pd.Timestamp("2011-01-02 06:00:00")
    if missing_pred_t in out.index:
        assert pd.isna(out.loc[missing_pred_t, "lag_1"])


def test_rolling_mean_excludes_current_target():
    df = load_raw_data()
    out = create_lag_features(df)
    
    # Check rolling mean 3h at t: it should be Mean(cnt[t-3], cnt[t-2], cnt[t-1]), strictly excluding cnt[t]
    t_test = pd.Timestamp("2011-01-01 04:00:00")
    if t_test in out.index:
        prior_vals = [
            df.loc[pd.Timestamp("2011-01-01 01:00:00"), "cnt"],
            df.loc[pd.Timestamp("2011-01-01 02:00:00"), "cnt"],
            df.loc[pd.Timestamp("2011-01-01 03:00:00"), "cnt"]
        ]
        expected_mean = np.mean(prior_vals)
        assert np.isclose(out.loc[t_test, "rolling_mean_3h"], expected_mean)
