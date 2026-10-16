from src.pipeline import generate_raw_data, clean_data, sql_analysis, build_features

def test_generated_data_has_expected_columns():
    raw = generate_raw_data()
    assert {"transaction_id", "date", "platform", "product", "price"}.issubset(raw.columns)
    assert raw["transaction_id"].duplicated().any()
    assert raw["price"].isna().any()

def test_cleaned_data_has_no_duplicate_product_day_keys():
    cleaned = clean_data(generate_raw_data())
    assert cleaned["price"].notna().all()
    assert (cleaned["price"] > 0).all()
    assert not cleaned.duplicated(["date", "platform", "product"]).any()

def test_sql_analysis_has_spike_columns():
    cleaned = clean_data(generate_raw_data())
    result = sql_analysis(cleaned)
    assert {"pct_change_1d", "pct_change_48h", "spike_48h"}.issubset(result.columns)

def test_feature_engineering_has_next_day_target():
    cleaned = clean_data(generate_raw_data())
    features = build_features(cleaned)
    assert "target_next_day_anomaly" in features.columns
    assert features["target_next_day_anomaly"].notna().all()
