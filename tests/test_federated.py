import pytest
import numpy as np
import flwr as fl
from unittest.mock import MagicMock
from src.config import Config
from src.federated.client import FraudClient, get_client_fn
from src.federated.server import TrackedFedAvg, get_on_fit_config_fn, get_evaluate_fn

def get_dummy_cfg():
    return Config(seed=42, n_customers_per_bank=5, transactions_per_customer=5, 
                  test_fraction=0.2, heterogeneity="default", hidden_dim=16, 
                  dropout=0.1, lr=0.001, local_epochs=1, batch_size=32, 
                  clients=4, rounds=1, aggregation="FedAvg", fedprox_mu=0.01, 
                  fraction_fit=1.0, global_eval=True, per_client_eval=True)

def test_flower_client_serialization():
    cfg = get_dummy_cfg()
    client = FraudClient(
        cid="test_cid",
        train_X=np.random.randn(10, 5).astype(np.float32),
        train_y=np.random.randint(0, 2, 10).astype(np.float32),
        test_X=np.random.randn(5, 5).astype(np.float32),
        test_y=np.random.randint(0, 2, 5).astype(np.float32),
        cfg=cfg,
        n_features=5
    )
    
    # Test get_parameters
    params = client.get_parameters({})
    assert isinstance(params, list)
    assert isinstance(params[0], np.ndarray)
    
    # Test set_parameters
    # Mutate a param
    params[0] = params[0] + 1.0
    client.set_parameters(params)
    new_params = client.get_parameters({})
    np.testing.assert_array_equal(params[0], new_params[0])

def test_flower_client_fit():
    cfg = get_dummy_cfg()
    client = FraudClient(
        cid="test_cid",
        train_X=np.random.randn(10, 5).astype(np.float32),
        train_y=np.random.randint(0, 2, 10).astype(np.float32),
        test_X=np.random.randn(5, 5).astype(np.float32),
        test_y=np.random.randint(0, 2, 5).astype(np.float32),
        cfg=cfg,
        n_features=5
    )
    
    params = client.get_parameters({})
    new_params, num_examples, metrics = client.fit(params, {"prox_mu": 0.0})
    
    assert num_examples == 10
    assert isinstance(new_params, list)
    
    # Check that parameters actually changed during training
    assert not np.array_equal(params[0], new_params[0])

def test_flower_client_evaluate():
    cfg = get_dummy_cfg()
    client = FraudClient(
        cid="test_cid",
        train_X=np.random.randn(10, 5).astype(np.float32),
        train_y=np.random.randint(0, 2, 10).astype(np.float32),
        test_X=np.random.randn(5, 5).astype(np.float32),
        test_y=np.random.randint(0, 2, 5).astype(np.float32),
        cfg=cfg,
        n_features=5
    )
    
    params = client.get_parameters({})
    loss, num_examples, metrics = client.evaluate(params, {})
    
    assert num_examples == 5
    assert "pr_auc" in metrics
    assert metrics["cid"] == "test_cid"

def test_tracked_fedavg_strategy():
    strategy = TrackedFedAvg()
    
    # Mock some evaluation results
    class MockProxy:
        def __init__(self, cid):
            self.cid = cid
            
    class MockEvalRes:
        def __init__(self, metrics):
            self.metrics = metrics
            self.loss = 0.0
            self.num_examples = 10
            
    results = [
        (MockProxy("bank_a"), MockEvalRes({"pr_auc": 0.8, "cid": "bank_a"})),
        (MockProxy("bank_b"), MockEvalRes({"pr_auc": 0.7, "cid": "bank_b"})),
    ]
    
    agg_loss, agg_metrics = strategy.aggregate_evaluate(1, results, [])
    
    # Check that TrackedFedAvg saved the per-client metrics properly
    history = strategy.client_metrics_history
    assert "round_1" in history
    assert "bank_a" in history["round_1"]
    assert "bank_b" in history["round_1"]
    assert history["round_1"]["bank_a"]["pr_auc"] == 0.8
    # Ensure CID is popped/omitted if configured, or exists
    assert "cid" not in history["round_1"]["bank_a"]

def test_fedprox_config_fn():
    fn = get_on_fit_config_fn(prox_mu=0.01)
    conf = fn(server_round=1)
    assert conf["prox_mu"] == 0.01
    assert conf["server_round"] == 1
