import pytest
import numpy as np
import pandas as pd
from src.config import load_config, Config
from src.data_generation import generate_federation
from src.federated.client import MaliciousFraudClient, get_client_fn
from src.federated.server import TrackedFedAvg, get_strategy
import flwr as fl

def test_iid_partitioning():
    cfg = load_config()
    cfg.heterogeneity = "iid"
    cfg.n_customers_per_bank = 10
    fed = generate_federation(cfg)
    
    assert len(fed) == 4
    for bank_id, df in fed.items():
        assert len(df["customer_id"].unique()) == 10

def test_unequal_client_sizes():
    cfg = load_config()
    cfg.client_sizes = [5, 10, 15, 20]
    fed = generate_federation(cfg)
    assert len(fed["bank_a"]["customer_id"].unique()) == 5
    assert len(fed["bank_b"]["customer_id"].unique()) == 10
    assert len(fed["bank_c"]["customer_id"].unique()) == 15
    assert len(fed["bank_d"]["customer_id"].unique()) == 20

def test_malicious_client_modification():
    cfg = load_config()
    cfg.local_epochs = 1
    
    # Dummy data
    X = np.random.randn(10, 5).astype(np.float32)
    y = np.random.randint(0, 2, size=(10,)).astype(np.float32)
    
    client = MaliciousFraudClient("test_bank", X, y, X, y, cfg, 5)
    
    # Mock parameters
    from src.model import FraudMLP, state_to_numpy
    model = FraudMLP(5, cfg.hidden_dim, cfg.dropout)
    params = [np.ones(p.shape, dtype=np.float32) for p in state_to_numpy(model)]
              
    new_params, num_examples, metrics = client.fit(params, {})
    
    # Since malicious client scales by -10.0, we check if they are very negative (since normal gradient updates on all 1s wouldn't do this)
    assert new_params[0][0][0] < -5.0
    
def test_dropout_configuration():
    cfg = load_config()
    cfg.fraction_fit = 0.5
    strategy = get_strategy(cfg, np.zeros((10,5)), np.zeros(10), 5)
    
    # Strategy should require minimum 2 clients to fit
    assert strategy.min_fit_clients == 2
