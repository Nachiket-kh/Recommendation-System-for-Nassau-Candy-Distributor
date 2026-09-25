"""Cost-aware, explainable shipping recommendations."""
from __future__ import annotations
import pandas as pd
from config import SCORING_WEIGHTS
from .model_training import FEATURES

def shipping_benchmarks(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby("Ship Mode").agg(average_cost=("Cost", "mean"), average_shipping_days=("shipping_days", "mean"), average_sales=("Sales", "mean"), average_gross_profit=("Gross Profit", "mean"), average_profit_margin=("profit_margin", "mean"), order_count=("Row ID", "count")).reset_index()

def recommend_shipping_mode(input_data: dict, model, history: pd.DataFrame) -> dict:
    """Rank actual modes using model likelihood and descriptive benchmarks; not causal savings."""
    row = pd.DataFrame([{f: input_data.get(f) for f in FEATURES}])
    probabilities = model.predict_proba(row)[0]
    candidates = shipping_benchmarks(history).set_index("Ship Mode")
    candidates["prediction_score"] = pd.Series(probabilities, index=model.classes_)
    for col, asc in [("average_cost", True), ("average_gross_profit", False), ("average_shipping_days", True)]:
        span = candidates[col].max() - candidates[col].min()
        candidates[col + "_score"] = 1.0 if span == 0 else ((candidates[col].max() - candidates[col]) if asc else (candidates[col] - candidates[col].min())) / span
    candidates["final_score"] = (SCORING_WEIGHTS["prediction"] * candidates["prediction_score"] + SCORING_WEIGHTS["cost"] * candidates["average_cost_score"] + SCORING_WEIGHTS["profit"] * candidates["average_gross_profit_score"] + SCORING_WEIGHTS["delivery"] * candidates["average_shipping_days_score"])
    ranked = candidates.sort_values("final_score", ascending=False).reset_index()
    best = ranked.iloc[0]
    return {"recommended_shipping_mode": best["Ship Mode"], "alternatives": ranked["Ship Mode"].iloc[1:3].tolist(), "model_confidence": float(best["prediction_score"]), "benchmarks": ranked.to_dict("records"), "reason": f"Highest transparent combined score: model probability {best['prediction_score']:.1%}, historical average cost {best['average_cost']:.2f}, and {best['order_count']:.0f} observed orders."}
