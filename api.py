
import os

import pandas as pd
import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv()

app = FastAPI(
    title="Retail Intelligence API",
    version="1.0.0",
    description="Retail price intelligence powered by PostgreSQL and synthetic data.",
)


def get_connection():
    return psycopg.connect(
        host=os.getenv("NIQ_DB_HOST", "127.0.0.1"),
        port=int(os.getenv("NIQ_DB_PORT", "5433")),
        dbname=os.getenv("NIQ_DB_NAME", "niq_retail"),
        user=os.getenv("NIQ_DB_USER", "niq_user"),
        password=os.getenv("NIQ_DB_PASSWORD"),
    )


@app.get("/health")
def health():
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return {"status": "ok", "data_source": "PostgreSQL"}
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Database connection failed. Check PostgreSQL and .env settings.",
        )


@app.get("/summary")
def summary():
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        COUNT(*) AS records,
                        COUNT(DISTINCT product) AS products,
                        COUNT(DISTINCT retailer) AS retailers
                    FROM retail_prices;
                """)
                row = cur.fetchone()

                cur.execute("""
                    WITH price_changes AS (
                        SELECT
                            product,
                            retailer,
                            date,
                            price,
                            LAG(price) OVER (
                                PARTITION BY product, retailer
                                ORDER BY date
                            ) AS previous_price
                        FROM retail_prices
                    )
                    SELECT COUNT(*)
                    FROM price_changes
                    WHERE previous_price IS NOT NULL
                      AND previous_price > 0
                      AND (price - previous_price) * 100.0
                          / previous_price > 20;
                """)
                spikes = cur.fetchone()[0]

        return {
            "records": row[0],
            "products": row[1],
            "retailers": row[2],
            "detected_48h_spikes": spikes,
        }
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve the summary from PostgreSQL.",
        )



@app.get("/prices")
def prices(product: str | None = None, platform: str | None = None):
    query = """
        SELECT product, retailer AS platform, category, date, price
        FROM retail_prices
        WHERE (%s::text IS NULL OR LOWER(product) = LOWER(%s::text))
          AND (%s::text IS NULL OR LOWER(retailer) = LOWER(%s::text))
        ORDER BY date DESC
        LIMIT 1000;
    """

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (product, product, platform, platform),
                )
                rows = cur.fetchall()
                columns = [desc.name for desc in cur.description]

        if not rows:
            raise HTTPException(
                status_code=404,
                detail="No matching records found.",
            )

        df = pd.DataFrame(rows, columns=columns)
        df["date"] = df["date"].astype(str)
        return df.to_dict(orient="records")

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Prices endpoint error: {exc!r}")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve prices from PostgreSQL.",
        )


@app.get("/anomalies")
def anomalies(min_change_pct: float = 20.0):
    if min_change_pct < 0:
        raise HTTPException(
            status_code=400,
            detail="min_change_pct must be non-negative.",
        )

    query = """
        WITH price_changes AS (
            SELECT
                product,
                retailer AS platform,
                category,
                date,
                price,
                LAG(price) OVER (
                    PARTITION BY product, retailer
                    ORDER BY date
                ) AS previous_price
            FROM retail_prices
        )
        SELECT
            product,
            platform,
            category,
            date,
            price,
            previous_price,
            ROUND(
                ((price - previous_price) * 100.0 / previous_price)::numeric,
                2
            ) AS pct_change_48h
        FROM price_changes
        WHERE previous_price IS NOT NULL
          AND previous_price > 0
          AND (price - previous_price) * 100.0
              / previous_price > %s
        ORDER BY pct_change_48h DESC
        LIMIT 1000;
    """

    try:
        with get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(min_change_pct,))
        return df.to_dict(orient="records")
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve anomalies from PostgreSQL.",
        )
