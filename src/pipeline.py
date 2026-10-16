from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

FESTIVAL_DATE = pd.Timestamp("2026-11-08")
START_DATE = pd.Timestamp("2026-10-01")
END_DATE = pd.Timestamp("2026-11-20")
PLATFORMS = ["Retailer_A", "Retailer_B", "Retailer_C"]
PRODUCTS = [
    ("Rice 5kg", "Staples", 320), ("Sunflower Oil 1L", "Cooking Oil", 145),
    ("Sugar 1kg", "Staples", 48), ("Toor Dal 1kg", "Pulses", 155),
    ("Wheat Flour 5kg", "Staples", 235), ("Tea 500g", "Beverages", 220),
    ("Coffee 200g", "Beverages", 165), ("Ghee 500ml", "Dairy", 310),
    ("Chocolates Box", "Festive Goods", 250), ("Dry Fruits 500g", "Festive Goods", 420),
    ("Detergent 1kg", "Home Care", 135), ("Biscuits Family Pack", "Packaged Food", 85),
]

def generate_raw_data(seed=42):
    """Generate synthetic retail data with deliberate data-quality issues."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    rows, txn = [], 1
    for product, category, base_price in PRODUCTS:
        for platform in PLATFORMS:
            factor = {"Retailer_A": 1.0, "Retailer_B": 1.025, "Retailer_C": .985}[platform]
            previous = base_price * factor
            for date in dates:
                price = previous * (1 + rng.normal(0, .012))
                days_to_festival = (FESTIVAL_DATE - date).days
                if 0 <= days_to_festival <= 2 and category in {"Festive Goods", "Beverages", "Staples"} and rng.random() < .42:
                    price *= rng.uniform(1.23, 1.42)
                if rng.random() < .035:
                    price *= rng.uniform(.88, .96)
                price = round(max(price, 1), 2)
                previous = price
                date_str = date.strftime("%d/%m/%Y") if txn % 3 == 0 else date.strftime("%Y-%m-%d")
                rows.append({
                    "transaction_id": f"T{txn:06d}", "date": date_str,
                    "platform": platform, "product": product, "category": category,
                    "price": price, "currency": "INR"
                })
                txn += 1
    raw = pd.DataFrame(rows)
    missing = rng.choice(raw.index, size=max(1, int(len(raw) * .025)), replace=False)
    raw.loc[missing, "price"] = np.nan
    duplicates = raw.sample(frac=.012, random_state=seed)
    raw = pd.concat([raw, duplicates], ignore_index=True)
    return raw.sample(frac=1, random_state=seed).reset_index(drop=True)

def clean_data(raw):
    """Normalize types, deduplicate and impute prices within product/platform series."""
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce", format="mixed")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "product", "platform", "category"])
    df = df.sort_values(["transaction_id"])
    df = df.drop_duplicates(subset=["transaction_id"], keep="first")
    df = df.drop_duplicates(subset=["date", "platform", "product"], keep="last")
    df = df.sort_values(["platform", "product", "date"])
    df["price_was_missing"] = df["price"].isna().astype(int)
    df["price"] = df.groupby(["platform", "product"])["price"].transform(
        lambda s: s.interpolate(limit_direction="both").ffill().bfill()
    )
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0].copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)

def sql_analysis(cleaned):
    """Use SQLite window functions to calculate price movements."""
    with sqlite3.connect(":memory:") as conn:
        cleaned.to_sql("retail_prices", conn, index=False, if_exists="replace")
        result = pd.read_sql_query("""
            WITH history AS (
              SELECT date, platform, product, category, price,
                LAG(price, 1) OVER (PARTITION BY platform, product ORDER BY date) AS price_1d,
                LAG(price, 2) OVER (PARTITION BY platform, product ORDER BY date) AS price_2d
              FROM retail_prices
            )
            SELECT *,
              ROUND(100.0 * (price-price_1d)/NULLIF(price_1d,0), 2) AS pct_change_1d,
              ROUND(100.0 * (price-price_2d)/NULLIF(price_2d,0), 2) AS pct_change_48h,
              CASE WHEN price_2d IS NOT NULL AND price > price_2d * 1.20 THEN 1 ELSE 0 END AS spike_48h
            FROM history ORDER BY date, platform, product
        """, conn)
    return result

def build_features(cleaned):
    df = cleaned.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["platform", "product", "date"]).reset_index(drop=True)
    group = df.groupby(["platform", "product"], group_keys=False)
    df["rolling_avg_price_3day"] = group["price"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["previous_day_price"] = group["price"].shift(1)
    df["previous_day_pct_change"] = 100 * (df["price"] - df["previous_day_price"]) / df["previous_day_price"].replace(0, np.nan)
    df["price_vs_rolling_avg_pct"] = 100 * (df["price"] - df["rolling_avg_price_3day"]) / df["rolling_avg_price_3day"].replace(0, np.nan)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["days_until_festival"] = (FESTIVAL_DATE - df["date"]).dt.days.clip(lower=0)
    df["is_festival_window"] = df["days_until_festival"].between(0, 3).astype(int)
    df["is_anomaly"] = ((df["rolling_avg_price_3day"].notna()) & (df["price"] > df["rolling_avg_price_3day"] * 1.20)).astype(int)
    df["target_next_day_anomaly"] = group["is_anomaly"].shift(-1)
    df = df.dropna(subset=["target_next_day_anomaly"]).copy()
    df["target_next_day_anomaly"] = df["target_next_day_anomaly"].astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in ["previous_day_pct_change", "price_vs_rolling_avg_pct"]:
        df[col] = df[col].fillna(0)
    df["rolling_avg_price_3day"] = df["rolling_avg_price_3day"].fillna(df["price"])
    return df

def train_model(features):
    """Train a simple baseline classifier; output metrics for honest evaluation."""
    categorical = ["platform", "product", "category"]
    numeric = ["price", "rolling_avg_price_3day", "previous_day_pct_change",
               "price_vs_rolling_avg_pct", "day_of_week", "days_until_festival", "is_festival_window"]
    data = features.sort_values("date").copy()
    cutoff = data["date"].quantile(.80)
    train, test = data[data["date"] <= cutoff], data[data["date"] > cutoff]
    if train.empty or test.empty or train["target_next_day_anomaly"].nunique() < 2:
        return None, "Not enough class variation to train a useful baseline on this dataset."
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric)
    ])
    model = Pipeline([
        ("prep", prep),
        ("model", RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=3,
                                         class_weight="balanced", random_state=42))
    ])
    model.fit(train[categorical + numeric], train["target_next_day_anomaly"])
    pred = model.predict(test[categorical + numeric])
    report = classification_report(test["target_next_day_anomaly"], pred, zero_division=0)
    features = features.copy()
    features["predicted_next_day_anomaly"] = model.predict(features[categorical + numeric])
    features.to_csv(OUTPUT_DIR / "model_predictions.csv", index=False)
    (OUTPUT_DIR / "model_evaluation.txt").write_text(report, encoding="utf-8")
    return model, report

def run_pipeline(seed=42):
    raw = generate_raw_data(seed)
    cleaned = clean_data(raw)
    sql = sql_analysis(cleaned)
    features = build_features(cleaned)
    raw.to_csv(OUTPUT_DIR / "raw_retail_data.csv", index=False)
    cleaned.to_csv(OUTPUT_DIR / "cleaned_retail_data.csv", index=False)
    sql.to_csv(OUTPUT_DIR / "sql_price_spikes.csv", index=False)
    features.to_csv(OUTPUT_DIR / "features.csv", index=False)
    model, report = train_model(features)
    return {"raw": raw, "cleaned": cleaned, "sql": sql, "features": features, "model": model, "report": report}
