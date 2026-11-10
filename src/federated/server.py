import numpy as np
import torch
import flwr as fl
from typing import Dict, Optional, Tuple, Callable

AGGREGATIONS = ("fedavg", "fedprox", "fedmedian", "fedtrimmedavg", "krum")


class _TrackingMixin:
    """Shared behaviour for all strategies: seeded client dropout, deterministic result
    ordering, and per-round bookkeeping of participants and per-client metrics."""

    def _init_tracking(self, seed: int = 0, dropout_rate: float = 0.0):
        self.client_metrics_history = {}
        self.participating_clients_history = {}
        self.latest_params = None
        self._seed = int(seed)
        self._dropout_rate = float(dropout_rate)

    def select_participants(self, ordered, server_round):
        """Seeded client dropout over a list ordered by client id. Always keeps >=1 survivor."""
        if self._dropout_rate <= 0 or not ordered:
            return list(ordered)
        rng = np.random.default_rng(self._seed * 1000 + server_round)
        keep = [x for x in ordered if rng.random() >= self._dropout_rate]
        if not keep:
            keep = [ordered[int(rng.integers(len(ordered)))]]
        return keep

    def configure_fit(self, server_round, parameters, client_manager):
        instructions = super().configure_fit(server_round, parameters, client_manager)
        ordered = sorted(instructions, key=lambda ins: int(ins[0].cid) if str(ins[0].cid).isdigit() else 0)
        return self.select_participants(ordered, server_round)

    def aggregate_fit(self, server_round: int, results, failures):
        # Fixed ordering -> floating-point reductions are reproducible regardless of Ray timing.
        results = sorted(results, key=lambda r: str(r[1].metrics.get("cid", r[0].cid)))
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
            round_metrics[cid] = {k: v for k, v in eval_res.metrics.items() if k != "cid"}
        self.client_metrics_history[f"round_{server_round}"] = round_metrics
        return agg_loss, agg_metrics


class TrackedFedAvg(_TrackingMixin, fl.server.strategy.FedAvg):
    def __init__(self, *args, seed: int = 0, dropout_rate: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_tracking(seed, dropout_rate)


class TrackedFedMedian(_TrackingMixin, fl.server.strategy.FedMedian):
    def __init__(self, *args, seed: int = 0, dropout_rate: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_tracking(seed, dropout_rate)


class TrackedFedTrimmedAvg(_TrackingMixin, fl.server.strategy.FedTrimmedAvg):
    def __init__(self, *args, seed: int = 0, dropout_rate: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_tracking(seed, dropout_rate)


class TrackedKrum(_TrackingMixin, fl.server.strategy.Krum):
    def __init__(self, *args, seed: int = 0, dropout_rate: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_tracking(seed, dropout_rate)


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
    from src.model import FraudMLP, state_to_numpy
    agg = cfg.aggregation.lower()
    if agg not in AGGREGATIONS:
        raise ValueError(f"Unknown aggregation '{cfg.aggregation}'. Expected one of {AGGREGATIONS}")

    # Seeded global initialisation (otherwise Flower asks a random client for its weights).
    torch.manual_seed(cfg.seed)
    init = state_to_numpy(FraudMLP(n_features, cfg.hidden_dim, cfg.dropout))

    common_args = dict(
        fraction_fit=cfg.fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=max(1, int(cfg.clients * cfg.fraction_fit)),
        min_evaluate_clients=cfg.clients,
        min_available_clients=cfg.clients,
        # Always provide a fit config so clients know the round number (used to seed training).
        on_fit_config_fn=get_on_fit_config_fn(cfg.fedprox_mu if agg == "fedprox" else 0.0),
        evaluate_fn=get_evaluate_fn(global_test_X, global_test_y, n_features, cfg),
        initial_parameters=fl.common.ndarrays_to_parameters(init),
        seed=cfg.seed,
        dropout_rate=cfg.dropout_rate,
    )
    if agg == "fedmedian":
        return TrackedFedMedian(**common_args)
    if agg == "fedtrimmedavg":
        return TrackedFedTrimmedAvg(beta=cfg.trim_beta, **common_args)
    if agg == "krum":
        return TrackedKrum(num_malicious_clients=max(1, len(cfg.malicious_set())),
                           num_clients_to_keep=0, **common_args)
    return TrackedFedAvg(**common_args)
