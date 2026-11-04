import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from analysis import load_and_audit, make_features, FEATURES


def test_dataset_audit_and_target_identity():
    raw, audit = load_and_audit()
    assert audit['rows'] == 17379
    assert audit['missing_cells'] == 0
    assert audit['duplicate_rows'] == 0
    assert audit['duplicate_timestamps'] == 0
    assert audit['missing_timestamps'] == 165
    assert audit['cnt_equals_casual_plus_registered'] is True


def test_features_do_not_include_target_components():
    raw, _ = load_and_audit()
    feat = make_features(raw)
    assert 'cnt' not in FEATURES
    assert 'casual' not in FEATURES
    assert 'registered' not in FEATURES
    assert set(FEATURES).issubset(feat.columns)


def test_lag_is_exact_elapsed_hour():
    raw, _ = load_and_audit()
    feat = make_features(raw)
    # At 2011-01-01 02:00, previous hour is present and should be 40.
    assert feat.loc[pd.Timestamp('2011-01-01 02:00:00'), 'lag_1h'] == 40
    # At 2011-01-02 06:00, the 05:00 timestamp is missing, so exact lag is missing.
    assert pd.isna(feat.loc[pd.Timestamp('2011-01-02 06:00:00'), 'lag_1h'])
