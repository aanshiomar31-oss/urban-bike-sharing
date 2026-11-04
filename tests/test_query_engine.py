"""Unit tests for the Natural-Language Query Understanding Engine across all 6 operational categories."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from src.query_engine import QueryEngine


@pytest.fixture(scope="module")
def engine():
    return QueryEngine()


def test_category_1_hourly_demand(engine):
    resp = engine.process_query("What are the busiest hours of the day?")
    assert resp["category"] == "Category 1: Hourly Demand"
    assert "direct_answer" in resp and len(resp["direct_answer"]) > 10
    assert "evidence" in resp and len(resp["evidence"]) > 5
    assert "interpretation" in resp
    assert "action" in resp
    assert "limitation" in resp
    assert resp["chart"] is not None


def test_category_2_monthly_demand(engine):
    resp = engine.process_query("Which month had the highest demand?")
    assert resp["category"] == "Category 2: Monthly Demand"
    assert "direct_answer" in resp
    assert "evidence" in resp


def test_category_3_weekday_weekend(engine):
    resp = engine.process_query("How does weekend demand differ from weekdays?")
    assert resp["category"] == "Category 3: Weekday and Weekend Analysis"
    assert "registered" in resp["direct_answer"].lower() or "commuter" in resp["direct_answer"].lower()


def test_category_4_forecasting_and_bias(engine):
    resp = engine.process_query("Why is the model underestimating peak demand?")
    assert resp["category"] == "Category 4: Forecasting Performance"
    assert "bias" in resp["evidence"].lower() or "peak" in resp["evidence"].lower()


def test_category_5_fleet_optimization(engine):
    resp = engine.process_query("How many active bikes might be needed for a demand of 500 rentals per hour?")
    assert resp["category"] == "Category 5: Fleet Optimization"
    assert "active bikes" in resp["direct_answer"].lower()


def test_category_6_business_recommendation(engine):
    resp = engine.process_query("Should we buy more bikes?")
    assert resp["category"] == "Category 6: Business Recommendations"
    assert "no" in resp["direct_answer"].lower()
