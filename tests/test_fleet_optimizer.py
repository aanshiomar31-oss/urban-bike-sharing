"""Unit tests for active fleet sensitivity formulas and scenarios."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import pandas as pd
from src.fleet_optimizer import calculate_active_fleet
from config import METRICS_DIR


def test_fleet_calculation_math():
    # 500 hourly demand, 0.5 hour trip (250 concurrent), 1.5 buffer (375), 10% maint -> 375 / 0.9 = 416.67
    val = calculate_active_fleet(hourly_demand=500.0, trip_duration_hours=0.5, spatial_buffer=1.5, maintenance_share=0.10)
    assert round(val, 2) == 416.67


def test_fleet_invalid_maintenance_share_raises():
    with pytest.raises(ValueError):
        calculate_active_fleet(500, 0.5, 1.5, maintenance_share=1.0)


def test_fleet_scenarios_generated():
    path = METRICS_DIR / "fleet_scenarios.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) > 0
    assert "estimated_fleet_needed" in df.columns
    assert (df["estimated_fleet_needed"] > 0).all()
