"""Fleet optimization and sensitivity modeling module.
Calculates active fleet requirements using operational queuing sensitivity formulas
across P50, P90, and P95 demand scenarios, trip turnaround durations, and spatial buffers.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from config import (
    METRICS_DIR,
    DEFAULT_TRIP_DURATION,
    DEFAULT_SPATIAL_BUFFER,
    DEFAULT_MAINTENANCE_SHARE,
    ILLUSTRATIVE_FLEET_SIZE,
    TARGET_COL,
)


def calculate_active_fleet(
    hourly_demand: float,
    trip_duration_hours: float = DEFAULT_TRIP_DURATION,
    spatial_buffer: float = DEFAULT_SPATIAL_BUFFER,
    maintenance_share: float = DEFAULT_MAINTENANCE_SHARE
) -> float:
    """Calculates active fleet requirement based on Little's Law and operational buffer overhead:
    Active Fleet = (Hourly Demand * Trip Duration * Spatial Buffer) / (1 - Maintenance Share)
    """
    if maintenance_share >= 1.0:
        raise ValueError("Maintenance share cannot be 100% or greater.")
    concurrent_active = hourly_demand * trip_duration_hours
    buffered_fleet = concurrent_active * spatial_buffer
    total_required = buffered_fleet / (1.0 - maintenance_share)
    return total_required


def generate_fleet_scenarios(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Generates sensitivity scenarios across Low (P50), Expected (Peak Mean), and High (P95) demand conditions."""
    p50_demand = float(df[TARGET_COL].quantile(0.50))
    p90_demand = float(df[TARGET_COL].quantile(0.90))
    p95_demand = float(df[TARGET_COL].quantile(0.95))
    peak_mean_demand = float(df[df.index.hour.isin([7, 8, 9, 17, 18, 19])][TARGET_COL].mean())

    demand_scenarios = {
        "Low Demand (Median / P50)": p50_demand,
        "Expected Demand (Commute Peak Mean)": peak_mean_demand,
        "Conservative Planning (P90)": p90_demand,
        "High Stress (P95 Observed)": p95_demand,
    }

    trip_durations = [0.4, 0.5, 0.6, 0.8]   # 24 mins to 48 mins
    spatial_buffers = [1.2, 1.5, 1.8, 2.0]  # Low to high spatial relocation friction
    maintenance_rates = [0.05, 0.10, 0.15]  # 5% to 15% out-of-service rate

    records = []
    for sc_name, d_val in demand_scenarios.items():
        for trip in trip_durations:
            for buf in spatial_buffers:
                for maint in maintenance_rates:
                    fleet_req = calculate_active_fleet(d_val, trip, buf, maint)
                    records.append({
                        "demand_scenario": sc_name,
                        "hourly_demand": round(d_val, 1),
                        "trip_duration_hours": trip,
                        "spatial_buffer": buf,
                        "maintenance_share": maint,
                        "estimated_fleet_needed": int(np.ceil(fleet_req)),
                        "fleet_2000_surplus_deficit": int(ILLUSTRATIVE_FLEET_SIZE - np.ceil(fleet_req))
                    })

    scenario_df = pd.DataFrame(records)
    scenario_df.to_csv(METRICS_DIR / "fleet_scenarios.csv", index=False)

    # Core baseline summary
    base_calc = calculate_active_fleet(p95_demand, DEFAULT_TRIP_DURATION, DEFAULT_SPATIAL_BUFFER, DEFAULT_MAINTENANCE_SHARE)
    fleet_summary = {
        "formula": "Fleet = (Hourly Demand * Trip Duration * Spatial Buffer) / (1 - Maintenance Share)",
        "observed_p50_demand": round(p50_demand, 1),
        "observed_peak_commute_mean": round(peak_mean_demand, 1),
        "observed_p90_demand": round(p90_demand, 1),
        "observed_p95_demand": round(p95_demand, 1),
        "default_assumptions": {
            "trip_duration_hours": DEFAULT_TRIP_DURATION,
            "spatial_buffer": DEFAULT_SPATIAL_BUFFER,
            "maintenance_share": DEFAULT_MAINTENANCE_SHARE,
            "illustrative_fleet_size": ILLUSTRATIVE_FLEET_SIZE
        },
        "baseline_p95_fleet_estimate": int(np.ceil(base_calc)),
        "baseline_p95_surplus_under_2000_bikes": int(ILLUSTRATIVE_FLEET_SIZE - np.ceil(base_calc)),
        "operational_takeaway": (
            f"Under standard assumptions (30-min average trip, 1.5x spatial buffer, 10% maintenance), "
            f"even a high-stress 95th-percentile hour ({p95_demand:.0f} rentals/hr) requires only "
            f"~{int(np.ceil(base_calc))} active bikes. A 2,000-bike fleet has a surplus of "
            f"{int(ILLUSTRATIVE_FLEET_SIZE - np.ceil(base_calc))} bikes on paper. "
            "Reported shortages are caused by spatial docking imbalance, not a fleet size deficit."
        )
    }

    with open(METRICS_DIR / "fleet_summary.json", "w") as f:
        json.dump(fleet_summary, f, indent=2)

    return fleet_summary, scenario_df
