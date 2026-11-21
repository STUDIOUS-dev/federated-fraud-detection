"""Tests for Phase E: communication accounting, update clipping, client-level DP, adaptive attacks."""
import math

import numpy as np
import pytest
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

from src.config import load_config
from src.experiments.framework import build_data, run_fl_inprocess
from src.experiments.study_e import build_scenarios
from src.experiments.study_e_report import render
from src.experiments.study_report import aggregate
from src.federated import privacy_robustness as pr
from src.federated.client import FraudClient, MaliciousFraudClient
from src.federated.server import get_strategy
from src.feature_engineering import BASE_FEATURES

N_FEAT = len(BASE_FEATURES)
OK = Status(Code.OK, "")


def tiny_cfg(**kw):
    cfg = load_config()
    cfg.n_customers_per_bank = 12
    cfg.transactions_per_customer = 10
    cfg.rounds = 2
    cfg.local_epochs = 1
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


class _Proxy:
    def __init__(self, cid):
        self.cid = cid


def _strategy(**kw):
    cfg = tiny_cfg(**kw)
    X = np.zeros((4, N_FEAT), np.float32)
    y = np.array([0, 1, 0, 1], np.float32)
    return cfg, get_strategy(cfg, X, y, N_FEAT)


def _results(strategy, deltas, sizes=(10, 20, 30, 40)):
    """Fake fit results whose parameters are global + delta_i (delta given as one flat vector)."""
    g = strategy._global
    gflat = pr.flatten(g)
    out = []
    for cid, d, n in zip(["bank_a", "bank_b", "bank_c", "bank_d"], deltas, sizes):
        params = ndarrays_to_parameters(pr.unflatten(gflat + d, g))
        out.append((_Proxy(cid), FitRes(status=OK, parameters=params, num_examples=n, metrics={"cid": cid})))
    return out


# ------------------------------------------------------------------ pure functions
def test_clip_update_bounds_norm_and_is_noop_below():
    v = np.array([3.0, 4.0])
    assert np.allclose(pr.clip_update(v, 1.0), [0.6, 0.8])
    assert np.allclose(pr.clip_update(v, 10.0), v)
    assert np.allclose(pr.clip_update(v, 0.0), v)


def test_zcdp_epsilon_behaviour():
    assert math.isinf(pr.zcdp_epsilon(0.0, 5))
    # more noise -> smaller epsilon; more rounds -> larger epsilon
    assert pr.zcdp_epsilon(2.0, 5) < pr.zcdp_epsilon(1.0, 5)
    assert pr.zcdp_epsilon(1.0, 10) > pr.zcdp_epsilon(1.0, 5)
    rho = 5 / 2.0
    assert pr.zcdp_epsilon(1.0, 5, 1e-5) == pytest.approx(rho + 2 * math.sqrt(rho * math.log(1e5)))


def test_alie_and_minmax_updates():
    rng = np.random.default_rng(0)
    benign = rng.normal(size=(3, 50))
    assert np.allclose(pr.alie_update(benign, 2.0), benign.mean(0) - 2.0 * benign.std(0))
    mal = pr.minmax_update(benign)
    pair = max(np.linalg.norm(a - b) for i, a in enumerate(benign) for b in benign[i + 1:])
    assert np.max(np.linalg.norm(benign - mal, axis=1)) <= pair + 1e-9
    assert np.linalg.norm(mal - benign.mean(0)) > 0  # it does move away from the mean
    assert pr.craft_adaptive("alie", benign[:1]) is None  # needs >= 2 honest updates
    with pytest.raises(ValueError):
        pr.craft_adaptive("nope", benign)


def test_unflatten_roundtrip():
    arrays = [np.arange(6, dtype=np.float32).reshape(2, 3), np.ones(4, np.float32)]
    back = pr.unflatten(pr.flatten(arrays), arrays)
    assert all(np.array_equal(a, b) and a.dtype == b.dtype for a, b in zip(arrays, back))


# ------------------------------------------------------------------ strategy behaviour
def test_get_strategy_validates_dp_settings():
    with pytest.raises(ValueError):
        _strategy(aggregation="fedmedian", clip_norm=0.5, dp_noise_multiplier=1.0)
    with pytest.raises(ValueError):
        _strategy(aggregation="fedavg", clip_norm=0.0, dp_noise_multiplier=1.0)
    _strategy(aggregation="fedmedian", clip_norm=0.5)  # clipping alone is fine with any aggregator


def test_communication_is_recorded_per_round():
    _, s = _strategy()
    n = pr.flatten(s._global).size
    res = _results(s, [np.zeros(n)] * 4)
    agg, _ = s.aggregate_fit(1, res, [])
    c = s.communication_history["round_1"]
    assert c["clients"] == 4
    assert c["upload_bytes"] == sum(pr.payload_bytes(r[1].parameters) for r in res)
    assert c["download_bytes"] == 4 * pr.payload_bytes(agg)
    assert c["upload_bytes"] > 4 * n * 4  # float32 payload plus serialization header


def test_default_strategy_is_unchanged_by_phase_e():
    """No clip / DP / adaptive attack -> plain weighted FedAvg, as in v1.1."""
    _, s = _strategy()
    n = pr.flatten(s._global).size
    rng = np.random.default_rng(1)
    deltas = [rng.normal(size=n) for _ in range(4)]
    g0 = pr.flatten(s._global)
    agg, _ = s.aggregate_fit(1, _results(s, deltas), [])
    w = np.array([10, 20, 30, 40]) / 100
    expected = g0 + sum(wi * d for wi, d in zip(w, deltas))
    assert np.allclose(pr.flatten(parameters_to_ndarrays(agg)), expected, atol=1e-5)
    assert s.update_norm_history == {}


def test_clipping_bounds_the_aggregate_step():
    _, s = _strategy(clip_norm=0.5)
    n = pr.flatten(s._global).size
    rng = np.random.default_rng(2)
    deltas = [rng.normal(size=n) * 10 for _ in range(4)]
    g0 = pr.flatten(s._global)
    agg, _ = s.aggregate_fit(1, _results(s, deltas), [])
    step = pr.flatten(parameters_to_ndarrays(agg)) - g0
    assert np.linalg.norm(step) <= 0.5 + 1e-4
    assert all(v > 0.5 for v in s.update_norm_history["round_1"].values())
    # the global model advances to the aggregate for the next round
    assert np.allclose(pr.flatten(s._global), pr.flatten(parameters_to_ndarrays(agg)))


def test_dp_noise_is_seeded_and_has_expected_scale():
    def run(seed):
        _, s = _strategy(clip_norm=0.5, dp_noise_multiplier=2.0, seed=seed)
        n = pr.flatten(s._global).size
        g0 = pr.flatten(s._global)
        agg, _ = s.aggregate_fit(1, _results(s, [np.zeros(n)] * 4), [])
        return pr.flatten(parameters_to_ndarrays(agg)) - g0

    a, b, c = run(7), run(7), run(8)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    expected_std = pr.gaussian_noise_std(2.0, 0.5, 4)  # 0.25
    assert a.std() == pytest.approx(expected_std, rel=0.15)


def test_dp_uses_uniform_weights():
    _, s = _strategy(clip_norm=100.0, dp_noise_multiplier=1e-9)
    n = pr.flatten(s._global).size
    deltas = [np.full(n, v) for v in (1.0, 0.0, 0.0, 0.0)]
    g0 = pr.flatten(s._global)
    agg, _ = s.aggregate_fit(1, _results(s, deltas, sizes=(1000, 1, 1, 1)), [])
    step = pr.flatten(parameters_to_ndarrays(agg)) - g0
    assert np.allclose(step, 0.25, atol=1e-4)  # not dominated by the 1000-example client


def test_adaptive_attack_replaces_only_malicious_updates():
    _, s = _strategy(attack="alie", malicious_clients=[2, 3], alie_z=1.0)
    n = pr.flatten(s._global).size
    rng = np.random.default_rng(3)
    deltas = [rng.normal(size=n) for _ in range(4)]
    out, norms = s._phase_e_transform(1, _results(s, deltas))
    g0 = pr.flatten(s._global)
    got = [pr.flatten(parameters_to_ndarrays(r[1].parameters)) - g0 for r in out]
    assert np.allclose(got[0], deltas[0], atol=1e-5) and np.allclose(got[1], deltas[1], atol=1e-5)
    expected = pr.alie_update(np.stack(deltas[:2]), 1.0)
    assert np.allclose(got[2], expected, atol=1e-5) and np.allclose(got[3], expected, atol=1e-5)


def test_non_adaptive_attack_does_not_enable_server_crafting():
    _, s = _strategy(attack="scale", malicious_clients=[3])
    assert s._adaptive_attack is None


def test_adaptive_malicious_client_trains_honestly():
    cfg = tiny_cfg(attack="minmax", malicious_clients=[0])
    cd, _, _ = build_data(cfg)
    d = cd["bank_a"]
    args = ("bank_a", d["train"][0], d["train"][1], d["test"][0], d["test"][1], cfg, N_FEAT)
    from src.model import FraudMLP, state_to_numpy
    init = state_to_numpy(FraudMLP(N_FEAT, cfg.hidden_dim, cfg.dropout))
    honest, _, _ = FraudClient(*args).fit(init, {"server_round": 1})
    mal, _, _ = MaliciousFraudClient(*args).fit(init, {"server_round": 1})
    assert all(np.array_equal(a, b) for a, b in zip(honest, mal))


# ------------------------------------------------------------------ end to end
@pytest.mark.parametrize("ov", [
    dict(aggregation="fedavg", clip_norm=0.5, dp_noise_multiplier=1.0),
    dict(aggregation="krum", attack="alie", malicious_clients=[2, 3], alie_z=3.0),
    dict(aggregation="fedmedian", attack="minmax", malicious_clients=[3], clip_norm=0.5),
])
def test_inprocess_runs_with_phase_e_settings(ov):
    cfg = tiny_cfg(**ov)
    cd, gt, _ = build_data(cfg)
    hist, s, _ = run_fl_inprocess(cfg, cd, gt)
    assert len(hist["pr_auc"]) == cfg.rounds + 1
    assert set(s.communication_history) == {"round_1", "round_2"}


def test_study_e_scenarios_and_report_render():
    sc = build_scenarios()
    assert {"central", "local", "fedavg", "dp_z1.0", "adaptive_minmax_x2_krum"} <= set(sc)
    for name, spec in sc.items():
        ov = spec["overrides"]
        if ov.get("dp_noise_multiplier"):
            assert ov.get("clip_norm", 0) > 0 and ov["aggregation"] == "fedavg", name

    def row(name, seed, pr_auc, **extra):
        r = {"scenario": name, "seed": seed, "kind": sc[name]["kind"], "overrides": sc[name]["overrides"],
             "global_final": {"pr_auc": pr_auc, "roc_auc": 0.8, "recall_at_fpr1": 0.3},
             "per_bank_final": {}, "n_test_pos": 50}
        r.update(extra)
        return r

    comm = {"communication": {"rounds": 5, "upload_bytes": 1000, "download_bytes": 1000}}
    runs = [row(n, s, 0.3 + 0.01 * s, **comm) for n in ("fedavg", "dp_z1.0", "fedavg_clip0.5") for s in (1, 2)]
    runs += [row("local", s, 0.29) for s in (1, 2)]
    md = render(aggregate(runs), runs)
    assert "Phase E Results" in md and "ε (δ=1e-5)" in md and "13.2" in md
