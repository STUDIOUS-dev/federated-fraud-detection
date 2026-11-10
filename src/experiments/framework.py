import os
import json
import time
import yaml
import random
from pathlib import Path
from dataclasses import asdict, replace
import torch
import numpy as np
import flwr as fl
import joblib
from src.config import Config, load_config
from src.data_generation import generate_federation
from src.feature_engineering import add_causal_features, BASE_FEATURES
from src.preprocessing import fit_scaler, temporal_split, to_xy

_SPLIT_CACHE: dict = {}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _bank_splits(cfg: Config) -> dict:
    """Per-bank (train, test) frames after causal feature engineering. Cached per data setting."""
    key = (cfg.seed, cfg.n_customers_per_bank, cfg.transactions_per_customer, cfg.test_fraction,
           cfg.heterogeneity, tuple(cfg.client_sizes) if cfg.client_sizes else None)
    if key not in _SPLIT_CACHE:
        banks = generate_federation(cfg)
        banks = {k: add_causal_features(v) for k, v in banks.items()}
        _SPLIT_CACHE[key] = {k: temporal_split(v, cfg.test_fraction) for k, v in banks.items()}
    return _SPLIT_CACHE[key]


def build_data(cfg: Config):
    """Returns (client_data, global_test, scaler).

    Training data follows the scenario (heterogeneity, client sizes). Test data ALWAYS comes from
    the reference federation (default heterogeneity, default sizes, same seed) so every scenario
    is scored on the identical held-out transactions and results are directly comparable.
    """
    train_splits = _bank_splits(cfg)
    ref_cfg = replace(cfg, heterogeneity="default", client_sizes=None)
    ref_splits = _bank_splits(ref_cfg)

    scaler = fit_scaler([tr for tr, _ in train_splits.values()])

    client_data = {}
    for bank_id in train_splits:
        tr_X, tr_y = to_xy(train_splits[bank_id][0], scaler)
        te_X, te_y = to_xy(ref_splits[bank_id][1], scaler)
        client_data[bank_id] = {"train": (tr_X, tr_y), "test": (te_X, te_y)}

    global_test = (np.vstack([v["test"][0] for v in client_data.values()]),
                   np.concatenate([v["test"][1] for v in client_data.values()]))
    return client_data, global_test, scaler


class ExperimentFramework:
    def __init__(self, exp_name: str, cfg: Config):
        self.exp_name = exp_name
        self.cfg = cfg
        self.results_dir = Path("results") / exp_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

        seed_everything(cfg.seed)

        with open(self.results_dir / "config.yaml", "w") as f:
            yaml.dump(asdict(cfg), f)

    def prepare_data(self):
        """Generates federated data, scales, and prepares Train/Test dicts."""
        client_data, global_test, scaler = build_data(self.cfg)
        joblib.dump(scaler, self.results_dir / "scaler.pkl")
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
            "fedprox_mu": self.cfg.fedprox_mu if self.cfg.aggregation.lower() == "fedprox" else None,
            "malicious_client_index": self.cfg.malicious_client_index,
            "malicious_clients": sorted(self.cfg.malicious_set()),
            "attack": self.cfg.attack if self.cfg.malicious_set() else None,
            "dropout_rate": self.cfg.dropout_rate,
            "heterogeneity": self.cfg.heterogeneity,
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


def run_fl_core(cfg: Config, client_data: dict, global_test: tuple):
    """Runs one Flower simulation. Returns (history, strategy, seconds). Writes nothing to disk."""
    from src.federated.client import get_client_fn
    from src.federated.server import get_strategy

    seed_everything(cfg.seed)
    n_features = len(BASE_FEATURES)
    client_fn = get_client_fn(client_data, cfg, n_features)
    strategy = get_strategy(cfg, global_test[0], global_test[1], n_features)

    start = time.time()
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.clients,
        config=fl.server.ServerConfig(num_rounds=cfg.rounds),
        strategy=strategy,
        ray_init_args={"ignore_reinit_error": True,
                       "num_cpus": int(os.environ.get("FL_NUM_CPUS", min(os.cpu_count() or 4, 12))),
                       "include_dashboard": False},
    )
    return history, strategy, time.time() - start


def run_fl_inprocess(cfg: Config, client_data: dict, global_test: tuple):
    """Ray-free equivalent of run_fl_core for large sweeps.

    Drives the SAME clients and the SAME Tracked* strategies through the same fit -> aggregate ->
    central-evaluate -> client-evaluate round protocol, in one process. tests/test_inprocess.py and the
    study runner verify it reproduces the Flower/Ray simulation. Returns (history_dict, strategy, seconds)
    where history_dict maps metric -> [(round, value), ...] (round 0 = initial model, as in Flower).
    """
    from flwr.common import (Code, Status, FitRes, EvaluateRes, ndarrays_to_parameters,
                             parameters_to_ndarrays)
    from src.federated.client import get_numpy_client_fn
    from src.federated.server import get_strategy

    assert cfg.fraction_fit == 1.0, "in-process runner models dropout via cfg.dropout_rate, not fraction_fit"
    seed_everything(cfg.seed)
    n_features = len(BASE_FEATURES)
    make = get_numpy_client_fn(client_data, cfg, n_features)
    strategy = get_strategy(cfg, global_test[0], global_test[1], n_features)
    ok = Status(Code.OK, "")

    class _Proxy:
        def __init__(self, cid): self.cid = str(cid)

    start = time.time()
    params = strategy.initial_parameters
    history: dict = {}

    def central_eval(rnd):
        out = strategy.evaluate(rnd, params)
        if out is not None:
            for k, v in out[1].items():
                history.setdefault(k, []).append((rnd, v))

    central_eval(0)
    for rnd in range(1, cfg.rounds + 1):
        fit_cfg = strategy.on_fit_config_fn(rnd) if strategy.on_fit_config_fn else {}
        nd_in = parameters_to_ndarrays(params)
        chosen = strategy.select_participants(list(range(cfg.clients)), rnd)
        fit_results = []
        for idx in chosen:
            new, n, m = make(idx).fit([a.copy() for a in nd_in], fit_cfg)
            fit_results.append((_Proxy(idx), FitRes(status=ok, parameters=ndarrays_to_parameters(new),
                                                    num_examples=n, metrics=m)))
        agg, _ = strategy.aggregate_fit(rnd, fit_results, [])
        if agg is not None:
            params = agg
        central_eval(rnd)
        nd = parameters_to_ndarrays(params)
        eval_results = []
        for idx in range(cfg.clients):
            loss, n, m = make(idx).evaluate(nd, {})
            eval_results.append((_Proxy(idx), EvaluateRes(status=ok, loss=loss, num_examples=n, metrics=m)))
        strategy.aggregate_evaluate(rnd, eval_results, [])
    return history, strategy, time.time() - start


def run_fl_experiment(exp_name: str, config_overrides: dict):
    cfg = load_config()
    for k, v in config_overrides.items():
        setattr(cfg, k, v)

    framework = ExperimentFramework(exp_name, cfg)
    client_data, global_test = framework.prepare_data()
    n_features = framework.get_n_features()

    history, strategy, execution_time = run_fl_core(cfg, client_data, global_test)

    results = {
        "exp_name": exp_name,
        "config": config_overrides,
        "history_centralized": history.metrics_centralized,
        "per_client_history": strategy.client_metrics_history,
        "participating_clients": strategy.participating_clients_history,
    }

    if strategy.latest_params is not None:
        from src.model import FraudMLP, numpy_to_state
        model = FraudMLP(n_features, cfg.hidden_dim, cfg.dropout)
        numpy_to_state(model, strategy.latest_params)
        torch.save(model.state_dict(), framework.results_dir / "model.pt")
        # Also save a sample of test instances for explainability
        np.save(framework.results_dir / "test_sample_X.npy", global_test[0][:100])
        np.save(framework.results_dir / "test_sample_y.npy", global_test[1][:100])

    framework.save_results(results, execution_time)
