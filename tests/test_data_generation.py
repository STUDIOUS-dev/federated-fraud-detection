import pandas as pd
import numpy as np
from src.data_generation import generate_bank_transactions, generate_federation
from src.config import Config

def test_generate_bank_transactions():
    cfg = Config(seed=42, n_customers_per_bank=5, transactions_per_customer=10, test_fraction=0.2, heterogeneity="default", hidden_dim=32, dropout=0.1, lr=0.001, local_epochs=2, batch_size=256, clients=4, rounds=5, aggregation="FedAvg", fedprox_mu=0.01, fraction_fit=1.0, global_eval=True, per_client_eval=True)
    df = generate_bank_transactions("bank_a", cfg)
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "bank_id" in df.columns
    assert "customer_id" in df.columns
    assert "amount" in df.columns
    assert "is_fraud" in df.columns
    assert "timestamp" in df.columns
    
    # Check that sorting is correct
    assert df["timestamp"].is_monotonic_increasing
    assert all(df["bank_id"] == "bank_a")

def test_generate_federation():
    cfg = Config(seed=42, n_customers_per_bank=2, transactions_per_customer=5, test_fraction=0.2, heterogeneity="default", hidden_dim=32, dropout=0.1, lr=0.001, local_epochs=2, batch_size=256, clients=4, rounds=5, aggregation="FedAvg", fedprox_mu=0.01, fraction_fit=1.0, global_eval=True, per_client_eval=True)
    federation = generate_federation(cfg)
    
    assert isinstance(federation, dict)
    assert len(federation) == 4 # 4 banks defined in config
    for bank_id, df in federation.items():
        assert not df.empty
        assert all(df["bank_id"] == bank_id)
