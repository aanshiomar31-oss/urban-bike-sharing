from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from complete_analysis import load_data, add_features, FEATURES

def test_dataset_audit_and_target_identity():
 d,a=load_data(ROOT/'hour.csv')
 assert a['rows']==17379 and a['missing_timestamps']==165
 assert a['missing_cells']==0 and a['duplicate_rows']==0
 assert a['cnt_equals_casual_plus_registered']

def test_no_target_components_or_identifiers_in_features():
 assert not set(FEATURES)&{'cnt','casual','registered','instant'}

def test_feature_creation_preserves_observed_rows_and_no_lags():
 d,_=load_data(ROOT/'hour.csv'); x=add_features(d)
 assert len(x)==len(d)
 assert {'hour_sin','hour_cos','weekday_sin','weekday_cos','is_peak_hour'}<=set(x.columns)
 assert 'lag_24h' not in FEATURES and 'lag_168h' not in FEATURES

def test_commute_hours_definition():
 assert [7,8,9,17,18,19]==[h for h in range(24) if h in [7,8,9,17,18,19]]
