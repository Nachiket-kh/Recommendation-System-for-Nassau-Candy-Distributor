"""Train and select measured shipping-mode classifiers."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from .preprocessing import build_preprocessor
from .evaluation import classification_metrics
from config import RANDOM_STATE, TEST_SIZE

NUMERIC_FEATURES = ["Units", "Sales", "Cost", "Gross Profit", "order_month", "order_quarter", "order_day_of_week"]
CATEGORICAL_FEATURES = ["Division", "Region", "Country/Region", "State/Province", "Product ID"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def train_shipping_model(data, output_dir: Path) -> tuple[Pipeline, dict]:
    x, y = data[FEATURES], data["Ship Mode"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    candidates = {"logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced"), "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE), "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE)}
    results, fitted = {}, {}
    for name, estimator in candidates.items():
        model = Pipeline([("preprocessor", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)), ("classifier", estimator)])
        model.fit(x_train, y_train)
        pred, proba = model.predict(x_test), model.predict_proba(x_test)
        metrics = classification_metrics(y_test, pred)
        top3 = model.classes_[proba.argsort(axis=1)[:, -min(3, len(model.classes_)):]]
        metrics["top_1_accuracy"] = metrics["accuracy"]
        metrics["top_3_accuracy"] = float(sum(label in row for label, row in zip(y_test, top3)) / len(y_test))
        results[name], fitted[name] = metrics, model
    chosen = max(results, key=lambda name: results[name]["f1_weighted"])
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[chosen], output_dir / "shipping_model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps({"selected_model": chosen, "models": results, "baseline_accuracy": float(y_test.value_counts().max() / len(y_test))}, indent=2))
    return fitted[chosen], {"selected_model": chosen, "models": results}
