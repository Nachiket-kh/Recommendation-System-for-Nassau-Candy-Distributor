"""EDA charts saved as reproducible static artifacts."""
from pathlib import Path
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "artifacts" / ".matplotlib"))
import matplotlib.pyplot as plt
import pandas as pd

def save_eda_plots(data: pd.DataFrame, output_dir: Path) -> None:
    """Save compact source-backed charts for core business dimensions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    charts = [("Sales", "Division", "sales_by_division"), ("Sales", "Region", "sales_by_region"), ("Gross Profit", "Division", "gross_profit_by_division"), ("Units", "Region", "units_by_region")]
    for metric, category, name in charts:
        series = data.groupby(category)[metric].sum().sort_values()
        ax = series.plot.barh(title=f"{metric} by {category}")
        ax.set_xlabel(metric); plt.tight_layout(); plt.savefig(output_dir / f"{name}.png", dpi=160); plt.close()
    ax = data.groupby("Ship Mode")["shipping_days"].mean().sort_values().plot.bar(title="Average shipping duration by mode")
    ax.set_ylabel("Days"); plt.tight_layout(); plt.savefig(output_dir / "shipping_days_by_mode.png", dpi=160); plt.close()
