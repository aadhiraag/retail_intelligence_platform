# FMCG Retail Intelligence Platform

A data analytics project that processes synthetic FMCG retail price data, stores cleaned records in PostgreSQL, visualizes price trends, and identifies potential price anomalies.

## Project Overview

This project demonstrates a retail analytics workflow using Python, SQL, PostgreSQL, FastAPI, and Streamlit. It simulates price observations across products and retailers to explore price movements and data quality.

**Note:** The dataset is synthetic and is not sourced from NIQ, NielsenIQ, or actual retailers.

## Key Features

- **Data processing:** Clean and prepare retail price data using Python and pandas.
- **Data quality checks:** Examine missing values, duplicates, and other data issues.
- **PostgreSQL integration:** Store and query cleaned retail price records.
- **Interactive dashboard:** Explore price trends using Streamlit and Plotly.
- **Price anomaly analysis:** Identify potential price spikes using SQL-based comparisons.
- **REST API:** Expose health, summary, price, and anomaly endpoints using FastAPI.
- **Containerization:** Run PostgreSQL using Docker Compose.

## Tech Stack

- **Languages:** Python, SQL
- **Data analysis:** pandas
- **Database:** PostgreSQL
- **API:** FastAPI, Uvicorn
- **Dashboard:** Streamlit, Plotly
- **Tools:** Docker, Git, GitHub, VS Code

## Architecture

1. Generate synthetic retail price data.
2. Clean and validate the data using the Python pipeline.
3. Load cleaned records into PostgreSQL.
4. Query price records and calculate potential anomalies.
5. Visualize price trends and analysis results in Streamlit.
6. Provide API access through FastAPI.

## Project Structure

```text
niq_retail_intelligence_platform/
├── app.py
├── api.py
├── load_to_postgres.py
├── docker-compose.yml
├── requirements.txt
├── README.md
├── src/
├── data/
├── outputs/
└── tests/
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd niq_retail_intelligence_platform
```

### 2. Create and activate a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file in the project root with the PostgreSQL connection settings required by the application.

Do not commit `.env` or publish database passwords.

### 5. Start PostgreSQL

```bash
docker compose up -d
```

Ensure the database credentials and port in your configuration match the Docker setup.

### 6. Load data into PostgreSQL

```bash
python load_to_postgres.py
```

### 7. Run the dashboard

```bash
streamlit run app.py
```

### 8. Run the API

Open another terminal with the virtual environment activated:

```bash
uvicorn api:app --reload --port 8000
```

Open `http://127.0.0.1:8000/docs` to explore the API endpoints.

## Testing

Run the available automated tests:

```bash
pytest -q
```

## Future Improvements

- Add automated data refresh and scheduled processing.
- Improve anomaly detection using statistical or machine learning methods.
- Add more comprehensive API and database tests.
- Evaluate the system against licensed real-world retail data.

## Disclaimer

This is an educational portfolio project using synthetic data. It is not affiliated with or endorsed by NIQ/NielsenIQ and does not represent actual retailer prices.
