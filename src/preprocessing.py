"""Reproducible cleaning and model preprocessing."""
from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates, retain valid source records, and flag invalid values as missing."""
    cleaned = data.drop_duplicates().copy()
    for column in ["Sales", "Units", "Cost"]:
        cleaned.loc[cleaned[column] <= 0, column] = pd.NA
    return cleaned

def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer([("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features),
                              ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_features)])
