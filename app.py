"""Streamlit dashboard for Nassau Candy shipping recommendations."""
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from config import DATA_PATH, MODEL_DIR
from src.data_loader import load_data, get_dataset_summary
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features
from src.recommendation import recommend_shipping_mode, shipping_benchmarks
from src.optimization import recommend_factory_and_shipping
from src.model_training import FEATURES

st.set_page_config(page_title="Nassau Candy Optimizer", layout="wide")
@st.cache_data
def get_data(): return engineer_features(clean_data(load_data(DATA_PATH)))
@st.cache_resource
def get_model(): return joblib.load(MODEL_DIR / "shipping_model.joblib")

try: data = get_data()
except Exception as exc: st.error(f"Unable to load dataset: {exc}"); st.stop()
st.title("Nassau Candy | Shipping Optimization")
tabs = st.tabs(["Overview", "Data Analysis", "Shipping Recommendation", "Optimization", "Model Performance", "Business Insights"])
with tabs[0]:
    metrics = {"Total Orders": data['Order ID'].nunique(), "Total Sales": data['Sales'].sum(), "Total Units": data['Units'].sum(), "Gross Profit": data['Gross Profit'].sum(), "Average Cost": data['Cost'].mean(), "Profit Margin": data['profit_margin'].mean()}
    cols = st.columns(6)
    for col, (name, value) in zip(cols, metrics.items()): col.metric(name, f"${value:,.2f}" if name in ["Total Sales", "Gross Profit", "Average Cost"] else (f"{value:.1%}" if name == "Profit Margin" else f"{value:,.0f}"))
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(px.bar(data.groupby('Division', as_index=False)['Sales'].sum(), x='Division', y='Sales', title='Sales by Division'), use_container_width=True)
    c2.plotly_chart(px.bar(data.groupby('Region', as_index=False)['Sales'].sum(), x='Region', y='Sales', title='Sales by Region'), use_container_width=True)
    c3.plotly_chart(px.histogram(data, x='Ship Mode', title='Shipping Mode Distribution'), use_container_width=True)
with tabs[1]:
    st.dataframe(data.head(100), use_container_width=True)
    st.subheader("Data quality")
    st.json(get_dataset_summary(data))
    st.plotly_chart(px.scatter(data, x='Sales', y='Cost', color='Ship Mode', hover_data=['Product Name'], title='Cost vs Sales'), use_container_width=True)
with tabs[2]:
    st.caption("Scores combine predicted likelihood with historical cost, profit, and delivery benchmarks. They are not causal estimates of savings.")
    left, right = st.columns(2)
    def choice(column): return left.selectbox(column, sorted(data[column].dropna().unique()))
    product_id = choice('Product ID'); division = right.selectbox('Division', sorted(data['Division'].unique())); region = right.selectbox('Region', sorted(data['Region'].unique()))
    country = right.selectbox('Country/Region', sorted(data['Country/Region'].unique())); state = right.selectbox('State/Province', sorted(data['State/Province'].unique()))
    units = left.number_input('Units', min_value=1, value=3); sales = left.number_input('Sales / Order Value', min_value=0.01, value=10.0); cost = left.number_input('Cost', min_value=0.01, value=3.0)
    if st.button('Recommend shipping mode'):
        try:
            model = get_model(); sample = data[data['Product ID'].eq(product_id)].iloc[0]
            inputs = {'Product ID': product_id, 'Division': division, 'Region': region, 'Country/Region': country, 'State/Province': state, 'Units': units, 'Sales': sales, 'Cost': cost, 'Gross Profit': max(sales-cost, 0), 'order_month': sample['order_month'], 'order_quarter': sample['order_quarter'], 'order_day_of_week': sample['order_day_of_week']}
            result = recommend_shipping_mode(inputs, model, data)
            st.success(f"Recommended: {result['recommended_shipping_mode']}"); st.write(f"Alternatives: {', '.join(result['alternatives']) or 'None'}"); st.write(f"Model confidence: {result['model_confidence']:.1%}"); st.info(result['reason'])
            st.dataframe(pd.DataFrame(result['benchmarks']), use_container_width=True)
        except FileNotFoundError: st.warning('Model files are missing. Run `python train.py` first.')
with tabs[3]:
    st.subheader('Factory allocation')
    st.info(recommend_factory_and_shipping({})['message'])
    st.dataframe(shipping_benchmarks(data), use_container_width=True)
with tabs[4]:
    try:
        report = json.loads((MODEL_DIR / 'metrics.json').read_text()); st.json(report)
    except FileNotFoundError: st.warning('No training metrics found. Run `python train.py`.')
with tabs[5]:
    top_division = data.groupby('Division')['Sales'].sum().idxmax(); top_region = data.groupby('Region')['Sales'].sum().idxmax(); mode = data['Ship Mode'].mode().iat[0]
    st.write(f"Highest-sales division: **{top_division}**. Highest-sales region: **{top_region}**. Most-used shipping mode: **{mode}**.")
    st.dataframe(data.groupby('Product Name', as_index=False).agg(Sales=('Sales','sum'), Units=('Units','sum'), Gross_Profit=('Gross Profit','sum')).sort_values('Sales', ascending=False).head(10), use_container_width=True)
