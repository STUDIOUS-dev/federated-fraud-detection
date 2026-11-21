import numpy as np
import torch
import flwr as fl
from typing import Dict, Optional, Tuple, Callable
from flwr.common import FitRes, ndarrays_to_parameters, parameters_to_ndarrays
from src.federated import privacy_robustness as pr

AGGREGATIONS = ("fedavg", "fedprox", "fedmedian", "fedtrimmedavg", "krum")


def _partition_index(proxy) -> int:
    """Bank index of a Flower client proxy.

    Under Ray, recent Flower versions set ``proxy.cid`` to a random node id and keep the partition
    (bank) index in ``proxy.partition_id``; older versions use the partition index as the cid.
    Ordering by the partition index makes the seeded dropout hit the same banks on every run.
    """
    pid = getattr(proxy, "partition_id", None)
    if pid is not None:
        return int(pid)
    cid = str(proxy.cid)
    return int(cid) if cid.isdigit() else 0


class _TrackingMixin:
    """Shared behaviour for all strategies: seeded client dropout, deterministic result
    ordering, and per-round bookkeeping of participants and per-client metrics."""

    def _init_tracking(self, seed: int = 0, dropout_rate: float = 0.0, clip_norm: float = 0.0,
                       noise_multiplier: float = 0.0, adaptive_attack: str | None = None,
                       malicious_cids=(), alie_z: float = 1.0):
        self.client_metrics_history = {}
        self.participating_clients_history = {}
        self.communication_history = {}
        self.update_norm_history = {}
        self.latest_params = None
        self._seed = int(seed)
        self._dropout_rate = float(dropout_rate)
        # Phase E (all off by default -> behaviour identical to v1.1)
        self._clip_norm = float(clip_norm)
        self._noise_multiplier = float(noise_multiplier)
        self._adaptive_attack = adaptive_attack
        self._malicious_cids = set(map(str, malicious_cids))
        self._alie_z = float(alie_z)
        # Global model the clients start the next round from (both runners aggregate from it).
        init = getattr(self, "initial_parameters", None)
        self._global = parameters_to_ndarrays(init) if init is not None else None

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
        ordered = sorted(instructions, key=lambda ins: _partition_index(ins[0]))
        return self.select_participants(ordered, server_round)

    @staticmethod
    def _cid(item) -> str:
        return str(item[1].metrics.get("cid", item[0].cid))

    def _phase_e_transform(self, server_round, results):
        """Adaptive attack (threat model) -> clipping (defence) -> uniform weights (DP).
        Returns the transformed results and the per-client update norms that were received."""
        g = self._global
        if g is None or not results:
            return results, {}
        deltas = {self._cid(r): pr.flatten(parameters_to_ndarrays(r[1].parameters)) - pr.flatten(g)
                  for r in results}

        if self._adaptive_attack and self._malicious_cids:
            benign = [d for c, d in deltas.items() if c not in self._malicious_cids]
            mal = pr.craft_adaptive(self._adaptive_attack, np.stack(benign), self._alie_z) if benign else None
            if mal is not None:
                for c in deltas:
                    if c in self._malicious_cids:
                        deltas[c] = mal.copy()  # colluding attackers send the same update

        norms = {c: float(np.linalg.norm(d)) for c, d in deltas.items()}
        if self._clip_norm > 0:
            deltas = {c: pr.clip_update(d, self._clip_norm) for c, d in deltas.items()}

        gflat = pr.flatten(g)
        out = []
        for proxy, res in results:
            c = self._cid((proxy, res))
            params = ndarrays_to_parameters(pr.unflatten(gflat + deltas[c], g))
            n = 1 if self._noise_multiplier > 0 else res.num_examples  # DP needs bounded weights
            out.append((proxy, FitRes(status=res.status, parameters=params, num_examples=n,
                                      metrics=res.metrics)))
        return out, norms

    def aggregate_fit(self, server_round: int, results, failures):
        # Fixed ordering -> floating-point reductions are reproducible regardless of Ray timing.
        results = sorted(results, key=self._cid)
        upload = sum(pr.payload_bytes(r[1].parameters) for r in results)
        phase_e = self._clip_norm > 0 or self._noise_multiplier > 0 or bool(self._adaptive_attack)
        if phase_e:
            results, norms = self._phase_e_transform(server_round, results)
            self.update_norm_history[f"round_{server_round}"] = norms
        agg_params, metrics = super().aggregate_fit(server_round, results, failures)
        if agg_params is not None and self._noise_multiplier > 0:
            rng = np.random.default_rng(self._seed * 7919 + server_round)
            std = pr.gaussian_noise_std(self._noise_multiplier, self._clip_norm, len(results))
            noisy = [(a + rng.normal(0.0, std, a.shape)).astype(np.float32)
                     for a in parameters_to_ndarrays(agg_params)]
            agg_params = ndarrays_to_parameters(noisy)
        if agg_params is not None:
            self.latest_params = parameters_to_ndarrays(agg_params)
            self._global = self.latest_params
        cids = [self._cid(r) for r in results]
        self.participating_clients_history[f"round_{server_round}"] = cids
        download = pr.payload_bytes(agg_params) * len(results) if agg_params is not None else 0
        self.communication_history[f"round_{server_round}"] = {
            "clients": len(results), "upload_bytes": int(upload), "download_bytes": int(download)}
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
    def __init__(self, *args, seed: int = 0, dropout_rate: float = 0.0, phase_e=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_tracking(seed, dropout_rate, **(phase_e or {}))


class TrackedFedMedian(_TrackingMixin, fl.server.strategy.FedMedian):
    def __init__(self, *args, seed: int = 0, dropout_rate: float = 0.0, phase_e=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_tracking(seed, dropout_rate, **(phase_e or {}))


class TrackedFedTrimmedAvg(_TrackingMixin, fl.server.strategy.FedTrimmedAvg):
    def __init__(self, *args, seed: int = 0, dropout_rate: float = 0.0, phase_e=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_tracking(seed, dropout_rate, **(phase_e or {}))


class TrackedKrum(_TrackingMixin, fl.server.strategy.Krum):
    def __init__(self, *args, seed: int = 0, dropout_rate: float = 0.0, phase_e=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_tracking(seed, dropout_rate, **(phase_e or {}))


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
    from src.config import BANK_PROFILES
    agg = cfg.aggregation.lower()
    if agg not in AGGREGATIONS:
        raise ValueError(f"Unknown aggregation '{cfg.aggregation}'. Expected one of {AGGREGATIONS}")

    noise = float(getattr(cfg, "dp_noise_multiplier", 0.0))
    clip = float(getattr(cfg, "clip_norm", 0.0))
    if noise > 0 and agg not in ("fedavg", "fedprox"):
        raise ValueError("Differential privacy noise is only calibrated for FedAvg/FedProx aggregation")
    if noise > 0 and clip <= 0:
        raise ValueError("dp_noise_multiplier > 0 requires clip_norm > 0 (the noise scale is relative to it)")
    attack = getattr(cfg, "attack", "scale")
    banks = list(BANK_PROFILES)
    phase_e = dict(
        clip_norm=clip, noise_multiplier=noise,
        adaptive_attack=attack if attack in pr.ADAPTIVE_ATTACKS and cfg.malicious_set() else None,
        malicious_cids=[banks[i] for i in sorted(cfg.malicious_set()) if i < len(banks)],
        alie_z=float(getattr(cfg, "alie_z", 1.0)),
    )

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
        phase_e=phase_e,
    )
    if agg == "fedmedian":
        return TrackedFedMedian(**common_args)
    if agg == "fedtrimmedavg":
        return TrackedFedTrimmedAvg(beta=cfg.trim_beta, **common_args)
    if agg == "krum":
        return TrackedKrum(num_malicious_clients=max(1, len(cfg.malicious_set())),
                           num_clients_to_keep=0, **common_args)
    return TrackedFedAvg(**common_args)
