import flwr as fl
from typing import Dict, List, Optional, Tuple, Callable
import json
from pathlib import Path

class TrackedFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_metrics_history = {}
        self.participating_clients_history = {}
        self.latest_params = None
        
    def aggregate_fit(self, server_round: int, results, failures):
        agg_params, metrics = super().aggregate_fit(server_round, results, failures)
        if agg_params is not None:
            self.latest_params = fl.common.parameters_to_ndarrays(agg_params)
        cids = [str(res.metrics.get("cid", client_proxy.cid)) for client_proxy, res in results]
        self.participating_clients_history[f"round_{server_round}"] = cids
        return agg_params, metrics
        
    def aggregate_evaluate(self, server_round: int, results, failures):
        agg_loss, agg_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        # Save per-client metrics
        round_metrics = {}
        for client_proxy, eval_res in results:
            cid = str(eval_res.metrics.get("cid", client_proxy.cid))
            # Filter out the cid from the actual stored metrics if desired
            cleaned_metrics = {k: v for k, v in eval_res.metrics.items() if k != "cid"}
            round_metrics[cid] = cleaned_metrics
            
        self.client_metrics_history[f"round_{server_round}"] = round_metrics
        return agg_loss, agg_metrics

class TrackedFedMedian(fl.server.strategy.FedMedian):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_metrics_history = {}
        self.participating_clients_history = {}
        self.latest_params = None
        
    def aggregate_fit(self, server_round: int, results, failures):
        agg_params, metrics = super().aggregate_fit(server_round, results, failures)
        if agg_params is not None:
            self.latest_params = fl.common.parameters_to_ndarrays(agg_params)
        cids = [str(res.metrics.get("cid", client_proxy.cid)) for client_proxy, res in results]
        self.participating_clients_history[f"round_{server_round}"] = cids
        return agg_params, metrics
        
    def aggregate_evaluate(self, server_round: int, results, failures):
        agg_loss, agg_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        round_metrics = {}
        for client_proxy, eval_res in results:
            cid = str(eval_res.metrics.get("cid", client_proxy.cid))
            cleaned_metrics = {k: v for k, v in eval_res.metrics.items() if k != "cid"}
            round_metrics[cid] = cleaned_metrics
            
        self.client_metrics_history[f"round_{server_round}"] = round_metrics
        return agg_loss, agg_metrics


def get_on_fit_config_fn(prox_mu: float) -> Callable[[int], Dict[str, fl.common.Scalar]]:
    def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
        return {"prox_mu": prox_mu, "server_round": server_round}
    return fit_config

def get_evaluate_fn(global_test_X, global_test_y, n_features, cfg) -> Callable:
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        if not cfg.global_eval:
            return None
            
        from src.model import FraudMLP, numpy_to_state, predict
        from src.evaluation import evaluate_probs
        
        model = FraudMLP(n_features, cfg.hidden_dim, cfg.dropout)
        numpy_to_state(model, parameters)
        
        metrics = evaluate_probs(global_test_y, predict(model, global_test_X))
        return 0.0, metrics
    return evaluate

def get_strategy(cfg, global_test_X, global_test_y, n_features):
    common_args = dict(
        fraction_fit=cfg.fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=max(1, int(cfg.clients * cfg.fraction_fit)),
        min_evaluate_clients=cfg.clients,
        min_available_clients=cfg.clients,
        on_fit_config_fn=get_on_fit_config_fn(cfg.fedprox_mu) if cfg.aggregation == "fedprox" else None,
        evaluate_fn=get_evaluate_fn(global_test_X, global_test_y, n_features, cfg),
    )
    if cfg.aggregation.lower() == "fedmedian":
        return TrackedFedMedian(**common_args)
    else:
        return TrackedFedAvg(**common_args)
