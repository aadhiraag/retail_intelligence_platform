
import psycopg
import pandas as pd
from src.pipeline import run_pipeline


DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5433,
    "dbname": "niq_retail",
    "user": "niq_user",
    "password": "RetailDev2026",
}


def main():
    result = run_pipeline()
    cleaned = result["cleaned"]

    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS retail_prices (
                    id BIGSERIAL PRIMARY KEY,
                    product TEXT NOT NULL,
                    retailer TEXT NOT NULL,
                    category TEXT,
                    date DATE NOT NULL,
                    price NUMERIC(12, 2) NOT NULL
                        CHECK (price > 0),
                    UNIQUE (product, retailer, date)
                );
            """)

            for _, row in cleaned.iterrows():
                cur.execute(
                    """
                    INSERT INTO retail_prices
                        (product, retailer, category, date, price)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (product, retailer, date)
                    DO UPDATE SET
                        category = EXCLUDED.category,
                        price = EXCLUDED.price;
                    """,
                    (
                        row["product"],
                        row["platform"],
                        row["category"],
                        pd.to_datetime(row["date"]).date(),
                        float(row["price"]),
                    ),
                )

            cur.execute("SELECT COUNT(*) FROM retail_prices;")
            count = cur.fetchone()[0]

    print(f"Successfully loaded data. Total database records: {count}")


if __name__ == "__main__":
    main()
