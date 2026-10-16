
import os

import pandas as pd
import plotly.express as px
import psycopg
import streamlit as st
from dotenv import load_dotenv
from src.pipeline import run_pipeline

load_dotenv()

st.set_page_config(
    page_title="Retail Intelligence",
    page_icon="🛒",
    layout="wide",
)

st.title("FMCG Retail Intelligence Platform")
st.caption(
    "Synthetic demonstration data only — "
    "not actual prices from any retailer."
)


@st.cache_data
def load_data():
    results = run_pipeline()
    return (
        results["raw"],
        results["cleaned"],
        results["sql"],
        results["features"],
        results["report"],
    )


@st.cache_data
def load_postgres_prices():
    with psycopg.connect(
        host=os.getenv("NIQ_DB_HOST", "127.0.0.1"),
        port=int(os.getenv("NIQ_DB_PORT", "5433")),
        dbname=os.getenv("NIQ_DB_NAME", "niq_retail"),
        user=os.getenv("NIQ_DB_USER", "niq_user"),
        password=os.getenv("NIQ_DB_PASSWORD"),
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    product,
                    retailer AS platform,
                    category,
                    date,
                    price
                FROM retail_prices
                ORDER BY date;
            """)
            rows = cur.fetchall()
            columns = [column.name for column in cur.description]

    df = pd.DataFrame(rows, columns=columns)
    df["date"] = pd.to_datetime(df["date"])
    return df


# Load pipeline results once
raw, cleaned, sql, features, report = load_data()

cleaned["date"] = pd.to_datetime(cleaned["date"])
sql["date"] = pd.to_datetime(sql["date"])

# Load PostgreSQL records
try:
    postgres_prices = load_postgres_prices()
except Exception as exc:
    st.error(
        "Could not connect to PostgreSQL. Check that Docker is running "
        "and your .env settings are correct."
    )
    st.caption(f"Technical details: {exc}")
    st.stop()

if postgres_prices.empty:
    st.warning("No retail price records were found in PostgreSQL.")
    st.stop()


# Sidebar filters use PostgreSQL data
st.sidebar.header("Filters")

platforms = st.sidebar.multiselect(
    "Retailer",
    sorted(postgres_prices["platform"].unique()),
    default=sorted(postgres_prices["platform"].unique()),
)

categories = st.sidebar.multiselect(
    "Category",
    sorted(postgres_prices["category"].unique()),
    default=sorted(postgres_prices["category"].unique()),
)

products = st.sidebar.multiselect(
    "Product",
    sorted(postgres_prices["product"].unique()),
    default=sorted(postgres_prices["product"].unique()),
)


# Filter price records from PostgreSQL
filtered = postgres_prices[
    postgres_prices["platform"].isin(platforms)
    & postgres_prices["category"].isin(categories)
    & postgres_prices["product"].isin(products)
]

# Existing pipeline-based anomaly calculations
filtered_sql = sql[
    sql["platform"].isin(platforms)
    & sql["category"].isin(categories)
    & sql["product"].isin(products)
]


# Summary metrics
c1, c2, c3, c4 = st.columns(4)

c1.metric("PostgreSQL records", f"{len(postgres_prices):,}")
c2.metric("Products", postgres_prices["product"].nunique())
c3.metric("Retailers", postgres_prices["platform"].nunique())
c4.metric("48-hour spikes", f"{int(sql['spike_48h'].sum()):,}")


# Price trends
st.subheader("Price trends")

if filtered.empty:
    st.info("Choose at least one filter value.")
else:
    chart = px.line(
        filtered,
        x="date",
        y="price",
        color="platform",
        line_group="product",
        hover_data=["product", "category"],
        title="Daily synthetic prices from PostgreSQL",
    )
    st.plotly_chart(chart, use_container_width=True)


# Price anomaly candidates
st.subheader("Price anomaly candidates")

anomalies = filtered_sql[
    filtered_sql["spike_48h"] == 1
].sort_values(
    "pct_change_48h",
    ascending=False,
)

st.dataframe(
    anomalies,
    use_container_width=True,
    hide_index=True,
)


# Data quality
st.subheader("Data quality")

quality = pd.DataFrame({
    "Metric": [
        "Raw rows",
        "Clean rows",
        "Duplicate transaction IDs in raw data",
        "Missing prices in raw data",
        "PostgreSQL price records",
    ],
    "Value": [
        len(raw),
        len(cleaned),
        int(raw["transaction_id"].duplicated().sum()),
        int(raw["price"].isna().sum()),
        len(postgres_prices),
    ],
})

st.dataframe(
    quality,
    use_container_width=True,
    hide_index=True,
)


# Model evaluation
st.subheader("Model evaluation")

st.code(
    report
    or "Model report unavailable; inspect outputs/model_evaluation.txt"
)

st.caption(
    "The model is a synthetic-data baseline. Validate against licensed "
    "real historical data before making business claims."
)
