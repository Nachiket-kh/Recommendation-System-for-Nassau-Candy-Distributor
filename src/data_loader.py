"""Dataset loading and validation utilities."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = ["Row ID", "Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID", "Country/Region", "City", "State/Province", "Postal Code", "Division", "Region", "Product ID", "Product Name", "Sales", "Units", "Gross Profit", "Cost"]
NUMERIC_COLUMNS = ["Row ID", "Customer ID", "Sales", "Units", "Gross Profit", "Cost"]
DATE_COLUMNS = ["Order Date", "Ship Date"]

def validate_schema(data: pd.DataFrame) -> dict:
    """Return schema diagnostics and raise for missing required columns."""
    missing = [c for c in EXPECTED_COLUMNS if c not in data.columns]
    unexpected = [c for c in data.columns if c not in EXPECTED_COLUMNS]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return {"missing_columns": missing, "unexpected_columns": unexpected}

def load_data(path: str | Path) -> pd.DataFrame:
    """Load source data, parse dates day-first, coerce known numerics, and derive shipping days."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = pd.read_csv(path)
    validate_schema(data)
    for col in NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in DATE_COLUMNS:
        data[col] = pd.to_datetime(data[col], dayfirst=True, errors="coerce")
    data["shipping_days"] = (data["Ship Date"] - data["Order Date"]).dt.days
    return data

def get_dataset_summary(data: pd.DataFrame) -> dict:
    """Summarize quality and date diagnostics without changing source values."""
    return {"rows": len(data), "columns": len(data.columns), "missing_values": data.isna().sum().to_dict(),
            "duplicate_rows": int(data.duplicated().sum()), "date_ranges": {c: (str(data[c].min().date()), str(data[c].max().date())) for c in DATE_COLUMNS},
            "shipping_days": data["shipping_days"].describe().to_dict(), "invalid_date_order": int((data["shipping_days"] < 0).sum())}
