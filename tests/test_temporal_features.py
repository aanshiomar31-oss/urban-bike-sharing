"""Unit tests for multi-resolution temporal features and cyclical encodings."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import numpy as np
import pandas as pd
from src.data_loader import load_raw_data
from src.temporal_features import create_temporal_features
from config import ALL_PEAK_HOURS


def test_cyclical_trigonometric_encodings_valid():
    df = load_raw_data().iloc[:500]
    out = create_temporal_features(df)
    
    # Values must be within [-1, 1]
    assert np.all(out["hour_sin"].between(-1.0, 1.0))
    assert np.all(out["hour_cos"].between(-1.0, 1.0))
    assert np.all(out["weekday_sin"].between(-1.0, 1.0))
    assert np.all(out["weekday_cos"].between(-1.0, 1.0))
    assert np.all(out["month_sin"].between(-1.0, 1.0))
    assert np.all(out["month_cos"].between(-1.0, 1.0))

    # Trigonometric identity: sin^2 + cos^2 = 1
    hour_circ = out["hour_sin"]**2 + out["hour_cos"]**2
    assert np.allclose(hour_circ, 1.0, atol=1e-5)


def test_commute_windows_flagged_correctly():
    df = load_raw_data().iloc[:100]
    out = create_temporal_features(df)
    for hr in range(24):
        matching = out[out["hour"] == hr]
        if not matching.empty:
            expected_peak = 1 if hr in ALL_PEAK_HOURS else 0
            assert (matching["is_peak_hour"] == expected_peak).all()
