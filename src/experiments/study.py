"""Multi-seed study (Phases B + C).

Runs every scenario below for several seeds and appends one JSON line per (scenario, seed) to
results/study/runs.jsonl. The runner is resumable: finished (scenario, seed) pairs are skipped.

    python -m src.experiments.study                       # all scenarios, seeds 42..46
    python -m src.experiments.study --only attack --seeds 42 43
    python -m src.experiments.study --report-only         # just rebuild aggregate.json / STUDY_RESULTS.md

Federated scenarios use the Ray-free in-process runner (identical clients/strategies, verified to
reproduce the Flower simulation bit-for-bit; see tests/test_inprocess.py).
"""
import argparse
import json
import re
import time
from dataclasses import replace
from pathlib import Path

from src.config import load_config
from src.experiments.framework import build_data, run_fl_inprocess, seed_everything
from src.experiments.runners import run_centralized, run_local
from src.feature_engineering import BASE_FEATURES

SEEDS = [42, 43, 44, 45, 46]
OUT_DIR = Path("results/study")
ATTACKER = [3]           # bank_d, same attacker as Exp 9/10
TWO_ATTACKERS = [2, 3]   # 50% of the federation
AGGS = ["fedavg", "fedmedian", "fedtrimmedavg", "krum"]
ATTACKS = ["scale", "signflip", "labelflip"]


def _fl(**ov):
    return {"kind": "fl", "overrides": ov}


def build_scenarios() -> dict:
    sc = {
        "central": {"kind": "central", "overrides": {}},
        "local": {"kind": "local", "overrides": {}},
        "fedavg": _fl(aggregation="fedavg"),
    }
    # FedProx proximal-strength sweep
    for mu in (0.01, 0.1, 1.0):
        sc[f"fedprox_mu{mu}"] = _fl(aggregation="fedprox", fedprox_mu=mu)
    # Distribution / size / imbalance (each trained on the scenario, scored on the SAME reference test pool)
    sc["central_iid"] = {"kind": "central", "overrides": {"heterogeneity": "iid"}}
    sc["local_iid"] = {"kind": "local", "overrides": {"heterogeneity": "iid"}}
    sc["fedavg_iid"] = _fl(aggregation="fedavg", heterogeneity="iid")
    sizes = [200, 50, 1000, 750]
    sc["local_sizes"] = {"kind": "local", "overrides": {"client_sizes": sizes}}
    sc["fedavg_sizes"] = _fl(aggregation="fedavg", client_sizes=sizes)
    imb = [10, 500, 500, 500]
    sc["local_imbalance"] = {"kind": "local", "overrides": {"client_sizes": imb}}
    sc["fedavg_imbalance"] = _fl(aggregation="fedavg", client_sizes=imb)
    # Genuine client dropout
    for p in (0.25, 0.5):
        sc[f"fedavg_dropout{p}"] = _fl(aggregation="fedavg", dropout_rate=p)
    # Robust aggregators without any attacker (cost of robustness)
    for a in AGGS[1:]:
        sc[f"{a}_clean"] = _fl(aggregation=a)
    # Attack x defence grid
    for attack in ATTACKS:
        for a in AGGS:
            sc[f"attack_{attack}_x1_{a}"] = _fl(aggregation=a, attack=attack, malicious_clients=ATTACKER)
    for attack in ("scale", "signflip"):
        for a in AGGS:
            sc[f"attack_{attack}_x2_{a}"] = _fl(aggregation=a, attack=attack, malicious_clients=TWO_ATTACKERS)
    return sc


def run_one(name: str, spec: dict, seed: int) -> dict:
    cfg = load_config()
    cfg = replace(cfg, seed=seed)
    for k, v in spec["overrides"].items():
        setattr(cfg, k, v)
    seed_everything(seed)
    client_data, global_test, _ = build_data(cfg)
    n_features = len(BASE_FEATURES)
    t0 = time.time()
    if spec["kind"] == "central":
        res = run_centralized(cfg, client_data, global_test, n_features)
        hist, per_client, participants = res["history_centralized"], res["per_client_history"], {}
    elif spec["kind"] == "local":
        res = run_local(cfg, client_data, global_test, n_features)
        hist, per_client, participants = res["history_centralized"], res["per_client_history"], {}
    else:
        hist, strat, _ = run_fl_inprocess(cfg, client_data, global_test)
        per_client, participants = strat.client_metrics_history, strat.participating_clients_history
    return {
        "scenario": name, "seed": seed, "kind": spec["kind"], "overrides": spec["overrides"],
        "global_final": {k: v[-1][1] for k, v in hist.items()},
        "pr_auc_by_round": [v for _, v in hist.get("pr_auc", [])],
        "per_bank_final": per_client.get(f"round_{cfg.rounds}", {}),
        "participants": participants,
        "n_test": int(len(global_test[1])), "n_test_pos": int(global_test[1].sum()),
        "seconds": round(time.time() - t0, 1),
    }


def load_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--only", default=None, help="regex on scenario name")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs_path = OUT_DIR / "runs.jsonl"
    if not args.report_only:
        done = {(r["scenario"], r["seed"]) for r in load_runs(runs_path)}
        todo = [(n, s, seed) for n, s in build_scenarios().items()
                if not args.only or re.search(args.only, n) for seed in args.seeds if (n, seed) not in done]
        print(f"{len(todo)} runs to do ({len(done)} already done)", flush=True)
        for i, (n, spec, seed) in enumerate(todo, 1):
            row = run_one(n, spec, seed)
            with open(runs_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{i}/{len(todo)}] {n} seed={seed} pr_auc={row['global_final'].get('pr_auc'):.4f} ({row['seconds']}s)", flush=True)
    from src.experiments.study_report import write_report
    write_report(runs_path)


if __name__ == "__main__":
    main()
