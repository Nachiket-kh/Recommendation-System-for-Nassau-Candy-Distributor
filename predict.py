"""Public prediction API used by Streamlit and scripts."""
import joblib
import pandas as pd
from config import MODEL_DIR
from src.recommendation import recommend_shipping_mode
from src.optimization import recommend_factory_and_shipping

def predict_shipping_mode(input_data: dict):
    return joblib.load(MODEL_DIR / "shipping_model.joblib").predict(pd.DataFrame([input_data]))[0]

__all__ = ["predict_shipping_mode", "recommend_shipping_mode", "recommend_factory_and_shipping"]
