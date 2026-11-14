"""Opt-in check that the Ray-free in-process runner reproduces the Flower/Ray simulation exactly.

Heavy (starts Ray), so it only runs when RUN_RAY_TESTS=1 and Ray is importable:

    RUN_RAY_TESTS=1 python -m pytest tests/test_inprocess.py -v       # optionally FL_NUM_CPUS=2

Needs `pip install "flwr[simulation]"`. Neither runner writes to disk.
"""
import math
import os

import numpy as np
import pytest

if os.environ.get("RUN_RAY_TESTS") != "1":
    pytest.skip("opt-in: set RUN_RAY_TESTS=1 to run the Flower/Ray equivalence test", allow_module_level=True)
ray = pytest.importorskip("ray", reason="Ray not installed (pip install 'flwr[simulation]')")

from src.config import load_config  # noqa: E402
from src.experiments.framework import build_data, run_fl_core, run_fl_inprocess  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _shutdown_ray():
    yield
    if ray.is_initialized():
        ray.shutdown()


def tiny_cfg(**kw):
    cfg = load_config()
    cfg.n_customers_per_bank = 12
    cfg.transactions_per_customer = 10
    cfg.rounds = 2
    cfg.local_epochs = 1
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _assert_tree_equal(a, b, path=""):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f"keys differ at {path}: {set(a)} vs {set(b)}"
        for k in a:
            _assert_tree_equal(a[k], b[k], f"{path}/{k}")
    else:
        assert _same(a, b), f"{path}: flower={a!r} inprocess={b!r}"


CASES = [
    pytest.param({}, id="fedavg"),
    pytest.param(
        dict(aggregation="fedmedian", malicious_clients=[3], attack="signflip", dropout_rate=0.5),
        id="fedmedian-signflip-dropout"),
]


@pytest.mark.parametrize("overrides", CASES)
def test_inprocess_matches_flower(overrides):
    cfg = tiny_cfg(**overrides)
    client_data, global_test, _ = build_data(cfg)

    history, s_fl, _ = run_fl_core(cfg, client_data, global_test)
    h_ip, s_ip, _ = run_fl_inprocess(cfg, client_data, global_test)

    fl_curve = history.metrics_centralized["pr_auc"]
    ip_curve = h_ip["pr_auc"]
    assert [r for r, _ in fl_curve] == [r for r, _ in ip_curve]
    for (r, a), (_, b) in zip(fl_curve, ip_curve):
        assert _same(float(a), float(b)), f"pr_auc round {r}: flower={a!r} inprocess={b!r}"

    assert s_fl.participating_clients_history == s_ip.participating_clients_history
    assert len(s_fl.latest_params) == len(s_ip.latest_params)
    for i, (a, b) in enumerate(zip(s_fl.latest_params, s_ip.latest_params)):
        np.testing.assert_array_equal(a, b, err_msg=f"final param tensor {i}")

    _assert_tree_equal(s_fl.client_metrics_history, s_ip.client_metrics_history, "client_metrics")
