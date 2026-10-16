from fastapi import FastAPI, HTTPException
from src.pipeline import run_pipeline

app = FastAPI(title="Retail Intelligence API", version="1.0.0",
              description="Demo API powered by synthetic retail data.")
results = run_pipeline()
cleaned = results["cleaned"]
sql_results = results["sql"]

@app.get("/health")
def health():
    return {"status": "ok", "data_source": "synthetic"}

@app.get("/summary")
def summary():
    return {
        "records": int(len(cleaned)),
        "products": int(cleaned["product"].nunique()),
        "retailers": int(cleaned["platform"].nunique()),
        "detected_48h_spikes": int(sql_results["spike_48h"].sum())
    }

@app.get("/prices")
def prices(product: str | None = None, platform: str | None = None):
    df = cleaned.copy()
    if product:
        df = df[df["product"].str.casefold() == product.casefold()]
    if platform:
        df = df[df["platform"].str.casefold() == platform.casefold()]
    if df.empty:
        raise HTTPException(status_code=404, detail="No matching synthetic records found.")
    return df.to_dict(orient="records")

@app.get("/anomalies")
def anomalies(min_change_pct: float = 20.0):
    df = sql_results[
        (sql_results["pct_change_48h"].notna()) &
        (sql_results["pct_change_48h"] > min_change_pct)
    ].copy()
    return df.to_dict(orient="records")
