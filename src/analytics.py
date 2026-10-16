
import sqlite3
import pandas as pd

from src.pipeline import clean_data, generate_raw_data, OUTPUT_DIR


def get_cleaned_data():
    raw = generate_raw_data()
    return clean_data(raw)


def retailer_price_comparison(cleaned):
    """Compare the latest observed price of each product by retailer."""
    with sqlite3.connect(":memory:") as conn:
        cleaned.to_sql("retail_prices", conn, index=False)

        query = """
        WITH ranked_prices AS (
            SELECT
                product,
                category,
                platform,
                price,
                date,
                ROW_NUMBER() OVER (
                    PARTITION BY product, platform
                    ORDER BY date DESC
                ) AS rn
            FROM retail_prices
        )
        SELECT
            product,
            category,
            platform,
            price AS latest_price,
            date AS latest_date
        FROM ranked_prices
        WHERE rn = 1
        ORDER BY product, latest_price;
        """

        return pd.read_sql_query(query, conn)


def rank_price_anomalies(cleaned):
    """Rank price changes over two daily intervals."""
    with sqlite3.connect(":memory:") as conn:
        cleaned.to_sql("retail_prices", conn, index=False)

        query = """
        WITH history AS (
            SELECT
                date,
                product,
                category,
                platform,
                price,
                LAG(price, 2) OVER (
                    PARTITION BY product, platform
                    ORDER BY date
                ) AS price_48h_ago
            FROM retail_prices
        ),
        changes AS (
            SELECT *,
                ROUND(
                    100.0 * (price - price_48h_ago)
                    / NULLIF(price_48h_ago, 0), 2
                ) AS change_pct
            FROM history
            WHERE price_48h_ago IS NOT NULL
        )
        SELECT *
        FROM changes
        WHERE change_pct > 20
        ORDER BY change_pct DESC;
        """

        return pd.read_sql_query(query, conn)


def calculate_basket_index(cleaned):
    """Calculate an equally weighted product-basket price index.

    Each product has equal weight. Retailer prices are averaged first,
    then the product prices are averaged for each date.
    The first date is indexed to 100.
    """
    with sqlite3.connect(":memory:") as conn:
        cleaned.to_sql("retail_prices", conn, index=False)

        query = """
        WITH product_prices AS (
            SELECT
                date,
                product,
                AVG(price) AS avg_product_price
            FROM retail_prices
            GROUP BY date, product
        ),
        daily_basket AS (
            SELECT
                date,
                AVG(avg_product_price) AS basket_price
            FROM product_prices
            GROUP BY date
        )
        SELECT
            date,
            ROUND(basket_price, 2) AS basket_price,
            ROUND(
                100.0 * basket_price
                / FIRST_VALUE(basket_price) OVER (ORDER BY date),
                2
            ) AS basket_price_index
        FROM daily_basket
        ORDER BY date;
        """

        return pd.read_sql_query(query, conn)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    data = get_cleaned_data()

    comparison = retailer_price_comparison(data)
    anomalies = rank_price_anomalies(data)
    basket = calculate_basket_index(data)

    comparison.to_csv(
        OUTPUT_DIR / "retailer_price_comparison.csv", index=False
    )
    anomalies.to_csv(
        OUTPUT_DIR / "ranked_price_anomalies.csv", index=False
    )
    basket.to_csv(
        OUTPUT_DIR / "basket_price_index.csv", index=False
    )

    print("\n--- RETAILER PRICE COMPARISON ---")
    print(comparison.head(9).to_string(index=False))

    print("\n--- TOP PRICE ANOMALIES ---")
    print(anomalies.head(10).to_string(index=False))

    print("\n--- BASKET PRICE INDEX ---")
    print(basket.head(5).to_string(index=False))
    print("\nLatest index:")
    print(basket.tail(1).to_string(index=False))
    print("\nAnalytics CSV files saved in:", OUTPUT_DIR)
