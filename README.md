# Factory Reallocation & Shipping Optimization for Nassau Candy

This project uses the supplied Nassau Candy distributor order-line CSV to analyze shipping choices and recommend a shipping mode. It is a reproducible Streamlit and Python project built from the actual data, not synthetic replacements.

## Dataset

`data/raw/Nassau Candy Distributor.csv` contains 10,194 rows and 18 source columns: Row ID, Order ID, Order Date, Ship Date, Ship Mode, Customer ID, Country/Region, City, State/Province, Postal Code, Division, Region, Product ID, Product Name, Sales, Units, Gross Profit, and Cost.

The source has no missing values or exact duplicate rows. Order dates span 2024-01-02 to 2025-12-31; ship dates span 2026-06-30 to 2030-06-28. Derived shipping duration ranges from 904 to 1,642 days. These implausibly large but non-negative durations are retained and reported as a source-data limitation, not silently corrected.

## What it does

- Validates schema, types, dates, missing values, and duplicate rows.
- Engineers per-row profitability, unit economics, ratios, calendar features, and `shipping_days`.
- Measures Logistic Regression, Random Forest, and HistGradientBoosting classifiers with stratified holdout data.
- Combines model probability with descriptive historical cost, profit, and delivery benchmarks using clearly configurable non-optimal weights in `config.py`.
- Provides an interactive six-tab Streamlit dashboard.

## Factory limitation

The source has no factory identifiers, locations, capacity, manufacturing cost, or compatibility information. Factory allocation therefore returns an explicit unavailable status rather than inventing factories. Add real `data/raw/factories.csv` data based on `data/raw/factories_template.csv` to extend the framework.

## Run

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
pytest -q
```

## Structure

`src/` contains loading, cleaning, features, training, evaluation, recommendations, and factory architecture. `predict.py` exposes the public prediction interfaces, while `tests/` contains core validation tests.

## Limitations

Shipping-mode labels describe historical choices, not a causal measure of optimal logistics. Historical cost and duration comparisons are benchmarks only; no future savings are claimed. The dataset’s delayed ship dates materially limit delivery conclusions.
