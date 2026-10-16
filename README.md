# FMCG Retail Intelligence Platform

A portfolio project combining data engineering, analytics, machine learning, API development and business intelligence. **All generated records and retailer labels are synthetic**; they are not real prices from Blinkit, Zepto, Swiggy Instamart or NIQ.

## Features
- Synthetic retail data generation with missing prices, duplicate transaction IDs and mixed date formats
- Pandas cleaning and validation
- SQLite SQL analytics using `LAG()` and price-change thresholds
- Rolling price features and a Random Forest next-day anomaly baseline
- Streamlit dashboard with filters and price trend charts
- FastAPI endpoints for health, summary, prices and anomaly candidates
- Pytest tests for key pipeline behavior

## Setup (Windows / VS Code)
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run the dashboard
```bash
streamlit run app.py
```

## Run the API
In a separate terminal:
```bash
uvicorn api:app --reload
```
Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## Run tests
```bash
pytest -q
```

## Run pipeline without dashboard
```bash
python -c "from src.pipeline import run_pipeline; r=run_pipeline(); print(r['report'])"
```

## Outputs
Generated CSVs and model report are written to `outputs/`.

## Modeling notes
- The synthetic target is based on a threshold rule against a prior rolling average.
- The model predicts the next day's rule-based anomaly using current-day features.
- A chronological holdout is used, but this is still a toy demonstration. Metrics may be optimistic because the data and labels are synthetic.
- The festival date is a simulation parameter, not an assertion about a real event calendar.
- The SQL two-day lag approximates 48 hours only because the generated series has one daily observation per product/platform.

## Suggested roadmap
1. Make the core pipeline and tests run.
2. Add PostgreSQL persistence and schema constraints.
3. Improve dashboard usability and add category-level basket index.
4. Add Docker packaging.
5. Add scheduled runs with Airflow only if time and environment allow.
6. Add an LLM-generated natural-language summary only after deterministic analytics are correct.

## Resume description (use only after implementing and verifying)
**FMCG Retail Intelligence Platform** | Python, Pandas, SQL, Scikit-learn, FastAPI, Streamlit
- Built a synthetic retail data pipeline to clean and validate product pricing records and identify unusual price changes using SQL window functions.
- Developed an interactive dashboard and REST API for price trends and anomaly summaries; evaluated a baseline next-day anomaly classifier using precision, recall and F1-score.
