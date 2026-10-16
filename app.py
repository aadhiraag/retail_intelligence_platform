import streamlit as st
import pandas as pd
import plotly.express as px
from src.pipeline import run_pipeline, OUTPUT_DIR

st.set_page_config(page_title="Retail Intelligence", page_icon="🛒", layout="wide")
st.title("FMCG Retail Intelligence Platform")
st.caption("Synthetic demonstration data only — not actual prices from any retailer.")

@st.cache_data
def load_data():
    results = run_pipeline()
    return (results["cleaned"], results["sql"], results["features"], results["report"])

cleaned, sql, features, report = load_data()
cleaned["date"] = pd.to_datetime(cleaned["date"])
sql["date"] = pd.to_datetime(sql["date"])

st.sidebar.header("Filters")
platforms = st.sidebar.multiselect("Retailer", sorted(cleaned["platform"].unique()), default=sorted(cleaned["platform"].unique()))
categories = st.sidebar.multiselect("Category", sorted(cleaned["category"].unique()), default=sorted(cleaned["category"].unique()))
products = st.sidebar.multiselect("Product", sorted(cleaned["product"].unique()), default=sorted(cleaned["product"].unique()))
filtered = cleaned[cleaned["platform"].isin(platforms) & cleaned["category"].isin(categories) & cleaned["product"].isin(products)]
filtered_sql = sql[sql["platform"].isin(platforms) & sql["category"].isin(categories) & sql["product"].isin(products)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clean records", f"{len(cleaned):,}")
c2.metric("Products", cleaned["product"].nunique())
c3.metric("Retailers", cleaned["platform"].nunique())
c4.metric("48-hour spikes", f"{int(sql['spike_48h'].sum()):,}")

st.subheader("Price trends")
if filtered.empty:
    st.info("Choose at least one filter value.")
else:
    chart = px.line(filtered, x="date", y="price", color="platform", line_group="product",
                    hover_data=["product", "category"], title="Daily synthetic prices")
    st.plotly_chart(chart, use_container_width=True)

st.subheader("Price anomaly candidates")
anomalies = filtered_sql[filtered_sql["spike_48h"] == 1].sort_values("pct_change_48h", ascending=False)
st.dataframe(anomalies, use_container_width=True, hide_index=True)

st.subheader("Data quality")
quality = pd.DataFrame({
    "Metric": ["Raw rows", "Clean rows", "Duplicate transaction IDs removed", "Missing prices in raw data"],
    "Value": [
        len(run_pipeline()["raw"]),
        len(cleaned),
        int(run_pipeline()["raw"]["transaction_id"].duplicated().sum()),
        int(run_pipeline()["raw"]["price"].isna().sum())
    ]
})
st.dataframe(quality, use_container_width=True, hide_index=True)

st.subheader("Model evaluation")
st.code(report or "Model report unavailable; inspect outputs/model_evaluation.txt")
st.caption("The model is a synthetic-data baseline. Validate against licensed real historical data before making business claims.")
