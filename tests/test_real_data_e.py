"""Phase E on real data (E5): scenario wiring, the n_features path of the in-process runner, and the report.
Runs on a tiny random dataset so the OpenML download is not needed."""
import json

import numpy as np
import pytest

from src.experiments import real_data_e
from src.experiments.real_data_e_report import write_report


def _tiny_data(n_features=7, seed=0):
    rng = np.random.default_rng(seed)

    def xy(n, fraud):
        X = rng.normal(size=(n, n_features)).astype(np.float32)
        y = (rng.random(n) < fraud).astype(np.float32)
        X[y == 1, 0] += 2.0  # learnable signal
        return X, y

    test = xy(300, 0.05)
    client_data = {b: {"train": xy(n, 0.05), "test": test}
                   for b, n in zip(real_data_e.BANK_IDS, (240, 160, 80, 60))}
    return client_data, test, n_features


def test_scenarios_are_well_formed():
    sc = real_data_e.build_scenarios()
    assert len(sc) == 24
    assert {"central", "local", "fedavg", "fedavg_clip", "dp_z1.0", "attack_scale_x1_fedmedian"} <= set(sc)
    for name, spec in sc.items():
        ov = spec["overrides"]
        if ov.get("dp_noise_multiplier"):
            assert ov["clip_norm"] == real_data_e.DP_CLIP and ov["aggregation"] == "fedavg"
        if name.startswith("adaptive_"):
            assert ov["malicious_clients"] == real_data_e.TWO_ATTACKERS and ov["attack"] in ("alie", "minmax")


@pytest.fixture
def tiny_runs(monkeypatch):
    monkeypatch.setattr(real_data_e, "ROUNDS", 2)
    sc = real_data_e.build_scenarios()
    data = _tiny_data()
    names = ["central", "local", "fedavg", "fedmedian_clean", "krum_clean", "fedavg_clip", "dp_z1.0",
             "attack_scale_x1_fedavg", "attack_scale_x1_fedavg_clip", "adaptive_alie_z5_x2_krum"]
    return [real_data_e.run_one(n, sc[n], seed, data) for n in names for seed in (0, 1)]


def test_run_one_uses_given_feature_count(tiny_runs):
    by = {(r["scenario"], r["seed"]): r for r in tiny_runs}
    for r in tiny_runs:
        assert len(r["pr_auc_by_round"]) == (2 if r["kind"] != "fl" else 3)  # FL also records round 0
        assert 0.0 <= r["global_final"]["pr_auc"] <= 1.0
        assert r["n_test"] == 300
    fed = by[("fedavg", 0)]
    assert fed["communication"]["rounds"] == 2 and fed["communication"]["upload_bytes"] > 0
    assert "privacy" in by[("dp_z1.0", 0)] and "privacy" not in fed
    assert set(by[("fedavg_clip", 0)]["update_norms"]) == {"round_1", "round_2"}
    # runs are seeded: the same scenario and seed give the same result
    again = real_data_e.run_one("fedavg", real_data_e.build_scenarios()["fedavg"], 0, _tiny_data())
    assert again["global_final"] == fed["global_final"]


def test_report_renders(tiny_runs, tmp_path, monkeypatch):
    monkeypatch.setattr(real_data_e, "ROUNDS", 2)
    runs_path = tmp_path / "runs.jsonl"
    runs_path.write_text("\n".join(json.dumps(r) for r in tiny_runs) + "\n")
    md = tmp_path / "report.md"
    write_report(runs_path, md)
    text = md.read_text()
    for section in ("## 1. Baselines", "## 2. Update clipping", "## 4. Adaptive attacks", "## 6. Paired"):
        assert section in text
    agg = json.loads((tmp_path / "aggregate.json").read_text())
    assert agg["scenarios"]["dp_z1.0"]["epsilon"] > 0
