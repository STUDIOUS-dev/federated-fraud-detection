import pandas as pd
from src.feature_engineering import add_causal_features

def test_current_transaction_is_not_in_customer_history():
    df = pd.DataFrame({
        "customer_id": ["c1", "c1"],
        "terminal_id": ["t1", "t1"],
        "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 01:00:00"], utc=True),
        "amount": [100.0, 200.0],
        "is_fraud": [0, 1],
        "bank_id": ["bank_a", "bank_a"],
    })
    out = add_causal_features(df)
    second = out.iloc[1]
    assert second["customer_mean_before"] == 100.0
    assert second["customer_count_before"] == 1
    assert second["terminal_count_before"] == 1
    assert second["is_new_terminal"] == 0
