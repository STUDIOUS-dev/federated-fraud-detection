import os
import json
import yaml
from pathlib import Path
from dataclasses import asdict
import torch
import numpy as np
import random
import flwr as fl
from src.config import Config, load_config
from src.data_generation import generate_federation
from src.feature_engineering import add_causal_features, BASE_FEATURES
from src.preprocessing import fit_scaler, temporal_split, to_xy
import joblib

class ExperimentFramework:
    def __init__(self, exp_name: str, cfg: Config):
        self.exp_name = exp_name
        self.cfg = cfg
        self.results_dir = Path("results") / exp_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seeds
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        
        # Save config
        with open(self.results_dir / "config.yaml", "w") as f:
            yaml.dump(asdict(cfg), f)

    def prepare_data(self):
        """Generates federated data, scales, and prepares Train/Test dicts."""
        banks = generate_federation(self.cfg)
        banks = {k: add_causal_features(v) for k, v in banks.items()}
        splits = {k: temporal_split(v, self.cfg.test_fraction) for k, v in banks.items()}
        
        scaler = fit_scaler([tr for tr, _ in splits.values()])
        joblib.dump(scaler, self.results_dir / "scaler.pkl")
        
        # Client specific data
        client_data = {}
        for bank_id, (tr, te) in splits.items():
            tr_X, tr_y = to_xy(tr, scaler)
            te_X, te_y = to_xy(te, scaler)
            client_data[bank_id] = {
                "train": (tr_X, tr_y),
                "test": (te_X, te_y)
            }
            
        # Global test pool for global evaluation
        test_X_all, test_y_all = [], []
        for v in client_data.values():
            test_X_all.append(v["test"][0])
            test_y_all.append(v["test"][1])
            
        global_test = (np.vstack(test_X_all), np.concatenate(test_y_all))
        
        return client_data, global_test

    def save_results(self, results: dict, execution_time: float = 0.0):
        """Saves results dictionary to JSON and appends to global summary."""
        with open(self.results_dir / "results.json", "w") as f:
            json.dump(results, f, indent=4)
            
        summary_path = Path("results/summary.jsonl")
        summary = {
            "exp_name": self.exp_name,
            "aggregation": self.cfg.aggregation,
            "seed": self.cfg.seed,
            "clients": self.cfg.clients,
            "rounds": self.cfg.rounds,
            "local_epochs": self.cfg.local_epochs,
            "lr": self.cfg.lr,
            "fedprox_mu": self.cfg.fedprox_mu if self.cfg.aggregation == "fedprox" else None,
            "malicious_client_index": self.cfg.malicious_client_index,
            "client_sizes": self.cfg.client_sizes,
            "participating_clients": results.get("participating_clients", {}),
            "global_metrics": {
                metric: values[-1][1] if values else None 
                for metric, values in results.get("history_centralized", {}).items()
            },
            "per_client_metrics": {
                k: v for k, v in results.get("per_client_history", {}).get(f"round_{self.cfg.rounds}", {}).items()
            },
            "execution_time": execution_time,
            "status": "success"
        }
        with open(summary_path, "a") as f:
            f.write(json.dumps(summary) + "\n")
            
    def get_n_features(self):
        return len(BASE_FEATURES)

def run_fl_experiment(exp_name: str, config_overrides: dict):
    from src.federated.client import get_client_fn
    from src.federated.server import get_strategy
    
    cfg = load_config()
    for k, v in config_overrides.items():
        setattr(cfg, k, v)
        
    framework = ExperimentFramework(exp_name, cfg)
    client_data, global_test = framework.prepare_data()
    n_features = framework.get_n_features()
    
    client_fn = get_client_fn(client_data, cfg, n_features)
    strategy = get_strategy(cfg, global_test[0], global_test[1], n_features)
    
    import time
    start_time = time.time()
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.clients,
        config=fl.server.ServerConfig(num_rounds=cfg.rounds),
        strategy=strategy,
        ray_init_args={"ignore_reinit_error": True, "num_cpus": 12},
    )
    
    execution_time = time.time() - start_time
    
    results = {
        "exp_name": exp_name,
        "config": config_overrides,
        "history_centralized": history.metrics_centralized,
        "per_client_history": strategy.client_metrics_history,
        "participating_clients": strategy.participating_clients_history,
    }
    
    if hasattr(strategy, 'latest_params') and strategy.latest_params is not None:
        from src.model import FraudMLP, numpy_to_state
        model = FraudMLP(n_features, cfg.hidden_dim, cfg.dropout)
        numpy_to_state(model, strategy.latest_params)
        torch.save(model.state_dict(), framework.results_dir / "model.pt")
        # Also save a sample of test instances for explainability
        np.save(framework.results_dir / "test_sample_X.npy", global_test[0][:100])
        np.save(framework.results_dir / "test_sample_y.npy", global_test[1][:100])
    
    framework.save_results(results, execution_time)
    print(f"Experiment {exp_name} complete. Results saved in {framework.results_dir}.")
    return results
