"""Tests for the Phase A-C changes: determinism, shared evaluation pool, dropout, attacks,
new aggregators, new metrics, the in-process runner and the study report."""
from dataclasses import replace

import numpy as np
import pytest

from src.config import load_config, BANK_PROFILES
from src.data_generation import generate_bank_transactions
from src.evaluation import evaluate_probs
from src.experiments.framework import build_data, run_fl_inprocess
from src.experiments.study_report import aggregate, paired
from src.federated.client import FraudClient, MaliciousFraudClient
from src.federated.local_training import train_local
from src.federated.server import (get_strategy, TrackedFedAvg, TrackedFedMedian,
                                  TrackedFedTrimmedAvg, TrackedKrum)
from src.model import FraudMLP, state_to_numpy


def tiny_cfg(**kw):
    cfg = load_config()
    cfg.n_customers_per_bank = 12
    cfg.transactions_per_customer = 10
    cfg.rounds = 2
    cfg.local_epochs = 1
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _toy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 5)).astype(np.float32)
    y = (rng.random(64) < 0.3).astype(np.float32)
    return X, y


def _params(n=5, hidden=8):
    return state_to_numpy(FraudMLP(n, hidden, 0.1))


def test_train_local_is_deterministic_given_seed():
    X, y = _toy()
    kw = dict(initial_state=_params(), X=X, y=y, n_features=5, hidden_dim=8, dropout=0.1,
              lr=0.01, local_epochs=2, batch_size=16)
    a, _ = train_local(seed=7, **kw)
    b, _ = train_local(seed=7, **kw)
    c, _ = train_local(seed=8, **kw)
    for x, y_ in zip(a, b):
        np.testing.assert_array_equal(x, y_)
    assert any(not np.array_equal(x, z) for x, z in zip(a, c))


def test_evaluate_probs_new_metrics():
    y = np.array([0] * 98 + [1] * 2, dtype=np.float32)
    perfect = np.array([0.1] * 98 + [0.9] * 2)
    m = evaluate_probs(y, perfect)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["recall_at_fpr1"] == pytest.approx(1.0)
    # no positives -> NaN rather than a misleading 0
    m0 = evaluate_probs(np.zeros(10, dtype=np.float32), np.linspace(0, 1, 10))
    assert np.isnan(m0["roc_auc"]) and np.isnan(m0["recall_at_fpr1"])


def test_iid_profile_is_mean_of_bank_profiles_and_changes_only_profile():
    cfg = tiny_cfg(heterogeneity="iid")
    df = generate_bank_transactions("bank_c", cfg)
    assert len(df["customer_id"].unique()) == cfg.n_customers_per_bank
    # all IID banks use the same profile: same customer-level amount scale on average
    means = [generate_bank_transactions(b, cfg)["amount"].median() for b in BANK_PROFILES]
    default_means = [generate_bank_transactions(b, replace(cfg, heterogeneity="default"))["amount"].median()
                     for b in BANK_PROFILES]
    assert np.std(np.log(means)) < np.std(np.log(default_means))


def test_all_scenarios_share_the_same_test_pool():
    base = tiny_cfg()
    _, gt_base, _ = build_data(base)
    for override in ({"heterogeneity": "iid"}, {"client_sizes": [4, 6, 20, 15]}):
        cfg = tiny_cfg(**override)
        cd, gt, _ = build_data(cfg)
        # scaler differs per scenario, so compare labels (identical transactions) and shapes
        np.testing.assert_array_equal(gt[1], gt_base[1])
        assert gt[0].shape == gt_base[0].shape


def test_dropout_selection_is_seeded_and_keeps_a_survivor():
    cfg = tiny_cfg(dropout_rate=0.9)
    strat = get_strategy(cfg, np.zeros((4, 14), dtype=np.float32), np.zeros(4), 14)
    for rnd in range(1, 30):
        a = strat.select_participants([0, 1, 2, 3], rnd)
        assert a == strat.select_participants([0, 1, 2, 3], rnd)
        assert len(a) >= 1
    sizes = {len(strat.select_participants([0, 1, 2, 3], r)) for r in range(1, 60)}
    assert min(sizes) < 4  # clients really do drop
    cfg0 = tiny_cfg(dropout_rate=0.0)
    s0 = get_strategy(cfg0, np.zeros((4, 14), dtype=np.float32), np.zeros(4), 14)
    assert s0.select_participants([0, 1, 2, 3], 1) == [0, 1, 2, 3]


@pytest.mark.parametrize("agg,cls", [("FedAvg", TrackedFedAvg), ("fedprox", TrackedFedAvg),
                                     ("FedMedian", TrackedFedMedian), ("fedtrimmedavg", TrackedFedTrimmedAvg),
                                     ("krum", TrackedKrum)])
def test_strategy_selection_is_case_insensitive(agg, cls):
    cfg = tiny_cfg(aggregation=agg)
    assert isinstance(get_strategy(cfg, np.zeros((4, 14), dtype=np.float32), np.zeros(4), 14), cls)


def test_fedprox_mu_only_forwarded_for_fedprox():
    z = (np.zeros((4, 14), dtype=np.float32), np.zeros(4), 14)
    assert get_strategy(tiny_cfg(aggregation="fedprox", fedprox_mu=0.3), *z).on_fit_config_fn(1)["prox_mu"] == 0.3
    assert get_strategy(tiny_cfg(aggregation="FedAvg", fedprox_mu=0.3), *z).on_fit_config_fn(1)["prox_mu"] == 0.0


def test_unknown_aggregation_raises():
    with pytest.raises(ValueError):
        get_strategy(tiny_cfg(aggregation="nope"), np.zeros((4, 14), dtype=np.float32), np.zeros(4), 14)


@pytest.mark.parametrize("attack", ["scale", "signflip", "labelflip"])
def test_attacks_differ_from_honest_update(attack):
    X, y = _toy()
    cfg = tiny_cfg(attack=attack, local_epochs=1, hidden_dim=8)
    params = _params()
    honest = FraudClient("bank_a", X, y, X, y, cfg, 5).fit(params, {"server_round": 1})[0]
    bad = MaliciousFraudClient("bank_a", X, y, X, y, cfg, 5).fit(params, {"server_round": 1})[0]
    assert any(not np.allclose(h, b) for h, b in zip(honest, bad))
    assert all(b.dtype == np.float32 for b in bad)


def test_signflip_reverses_update_direction():
    X, y = _toy()
    cfg = tiny_cfg(attack="signflip", hidden_dim=8)
    params = _params()
    honest = FraudClient("bank_a", X, y, X, y, cfg, 5).fit(params, {"server_round": 1})[0]
    bad = MaliciousFraudClient("bank_a", X, y, X, y, cfg, 5).fit(params, {"server_round": 1})[0]
    d_h = np.concatenate([(h - p).ravel() for h, p in zip(honest, params)])
    d_b = np.concatenate([(b - p).ravel() for b, p in zip(bad, params)])
    np.testing.assert_allclose(d_b, -5.0 * d_h, rtol=1e-3, atol=1e-5)


def test_malicious_set_back_compat():
    cfg = tiny_cfg()
    assert cfg.malicious_set() == set()
    cfg.malicious_client_index = 3
    assert cfg.malicious_set() == {3}
    cfg.malicious_clients = [1, 2]
    assert cfg.malicious_set() == {1, 2}


def test_inprocess_runner_is_deterministic_and_tracks_clients():
    cfg = tiny_cfg(aggregation="fedmedian", malicious_clients=[3], attack="signflip", dropout_rate=0.5)
    cd, gt, _ = build_data(cfg)
    h1, s1, _ = run_fl_inprocess(cfg, cd, gt)
    h2, s2, _ = run_fl_inprocess(cfg, cd, gt)
    assert h1["pr_auc"] == h2["pr_auc"] or all(
        (np.isnan(a[1]) and np.isnan(b[1])) or a == b for a, b in zip(h1["pr_auc"], h2["pr_auc"]))
    assert [r for r, _ in h1["pr_auc"]] == [0, 1, 2]
    assert s1.participating_clients_history == s2.participating_clients_history
    assert set(s1.client_metrics_history["round_2"]) == set(BANK_PROFILES)


def test_study_report_aggregate_and_paired():
    def run(s, name, v):
        return {"scenario": name, "seed": s, "kind": "fl", "overrides": {}, "n_test_pos": 50,
                "global_final": {"pr_auc": v}, "per_bank_final": {"bank_a": {"pr_auc": v}}}
    runs = [run(1, "a", 0.30), run(2, "a", 0.32), run(3, "a", 0.28),
            run(1, "b", 0.25), run(2, "b", 0.26), run(3, "b", 0.24)]
    agg = aggregate(runs)
    assert agg["a"]["global"]["pr_auc"]["mean"] == pytest.approx(0.30)
    p = paired(runs, "a", "b")
    assert p["n"] == 3 and p["wins"] == 3 and p["mean_diff"] == pytest.approx(0.05)


def test_dropout_order_uses_partition_index_not_random_cid():
    """Under Ray, Flower proxies carry a random node id as cid; dropout must follow the bank index."""
    from src.federated.server import _partition_index

    class RayLikeProxy:
        def __init__(self, cid, partition_id):
            self.cid, self.partition_id = cid, partition_id

    class LegacyProxy:
        def __init__(self, cid):
            self.cid = cid

    ray_like = [RayLikeProxy("14237238010823199150", 2), RayLikeProxy("5389875831426782007", 0),
                RayLikeProxy("285298315600700548", 1), RayLikeProxy("13678377834088640357", 3)]
    assert [p.partition_id for p in sorted(ray_like, key=_partition_index)] == [0, 1, 2, 3]
    assert [_partition_index(LegacyProxy(c)) for c in ("0", "3", "bank_x")] == [0, 3, 0]
