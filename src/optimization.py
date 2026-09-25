"""Extensible factory allocation interface."""
from __future__ import annotations
import pandas as pd

def recommend_factory_and_shipping(order_data: dict, factory_data: pd.DataFrame | None = None) -> dict:
    if factory_data is None or factory_data.empty:
        return {"factory_available": False, "message": "Factory allocation requires a factory configuration dataset. The current dataset contains order, product, cost and shipping information but no factory-level attributes."}
    required = {"Factory ID", "Factory Name", "Capacity", "Variable Cost"}
    missing = sorted(required - set(factory_data.columns))
    if missing:
        return {"factory_available": False, "message": f"Factory data is incomplete; missing: {', '.join(missing)}."}
    return {"factory_available": False, "message": "Factory scoring needs real compatibility, capacity usage, distance, and logistics-cost data; no unsupported allocation was generated."}
