"""Feature engineering using only order-time attributes."""
from __future__ import annotations
import numpy as np
import pandas as pd

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create transparent per-row features; no target-derived aggregates are used."""
    result = data.copy()
    result["profit_margin"] = np.where(result["Sales"] != 0, result["Gross Profit"] / result["Sales"], np.nan)
    for name, numerator in [("cost_per_unit", "Cost"), ("sales_per_unit", "Sales"), ("gross_profit_per_unit", "Gross Profit")]:
        result[name] = np.where(result["Units"] != 0, result[numerator] / result["Units"], np.nan)
    result["cost_to_sales_ratio"] = np.where(result["Sales"] != 0, result["Cost"] / result["Sales"], np.nan)
    result["units_to_sales_ratio"] = np.where(result["Sales"] != 0, result["Units"] / result["Sales"], np.nan)
    result["order_month"] = result["Order Date"].dt.month
    result["order_year"] = result["Order Date"].dt.year
    result["order_quarter"] = result["Order Date"].dt.quarter
    result["order_day_of_week"] = result["Order Date"].dt.dayofweek
    return result
