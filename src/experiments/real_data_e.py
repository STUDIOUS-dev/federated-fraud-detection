"""Phase E on real data (E5): clipping, client-level DP, scaling and adaptive attacks on the OpenML
credit-card dataset (id 1597), using the SAME clients and Tracked* strategies as the synthetic study.

Separate from both the frozen synthetic results and the frozen real-data study
(`results/real_data/`, `docs/REAL_DATA_RESULTS.md`, which uses its own aggregation loop). Writes only
results/real_data_e/runs.jsonl, results/real_data_e/aggregate.json and docs/REAL_DATA_PHASE_E_RESULTS.md.

Data and banks are exactly those of `real_data_study.py`: temporal split by row order (80/20), the
training rows randomly assigned to 4 banks with unequal sizes and fraud shares (bank partition seeded
by the run seed), features standardised on the training split. Every bank is scored on the shared
test split. Resumable, like the other studies.

    python -m src.experiments.real_data_e --workers 4
    python -m src.experiments.real_data_e --only adaptive --seeds 0
    python -m src.experiments.real_data_e --report-only
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.config import load_config
from src.experiments.study import _phase_e_fields, load_runs

OUT_DIR = Path("results/real_data_e")
MD_PATH = Path("docs/REAL_DATA_PHASE_E_RESULTS.md")
SEEDS = [0, 1, 2, 3, 4]          # the seeds of the frozen real-data study
ROUNDS = 10                      # as in the frozen real-data study
BANK_IDS = ["bank_a", "bank_b", "bank_c", "bank_d"]  # 45% / 30% / 15% / 10% of the training rows
ATTACKER = [0]                   # largest bank, as in the frozen real-data study
TWO_ATTACKERS = [2, 3]           # the two smallest banks (25% of the rows), 50% of the federation
DP_CLIP = 1.0                    # ~ median honest update norm over all rounds (2-4 in round 1, 0.5-1.5 later)
DP_NOISE = (0.5, 1.0, 2.0)
AGGS = ["fedavg", "fedmedian", "krum"]
ADAPTIVE = {"alie": {"attack": "alie", "alie_z": 1.0},
            "alie_z3": {"attack": "alie", "alie_z": 3.0},
            "alie_z5": {"attack": "alie", "alie_z": 5.0},
            "minmax": {"attack": "minmax"}}


def _fl(**ov):
    return {"kind": "fl", "overrides": ov}


def build_scenarios() -> dict:
    sc = {
        "central": {"kind": "central", "overrides": {}},
        "local": {"kind": "local", "overrides": {}},
        "fedavg": _fl(aggregation="fedavg"),
        "fedmedian_clean": _fl(aggregation="fedmedian"),
        "krum_clean": _fl(aggregation="krum"),
        "fedavg_clip": _fl(aggregation="fedavg", clip_norm=DP_CLIP),
    }
    for z in DP_NOISE:
        sc[f"dp_z{z}"] = _fl(aggregation="fedavg", clip_norm=DP_CLIP, dp_noise_multiplier=z)
    sc["attack_scale_x1_fedavg"] = _fl(aggregation="fedavg", attack="scale", malicious_clients=ATTACKER)
    sc["attack_scale_x1_fedavg_clip"] = _fl(aggregation="fedavg", attack="scale", malicious_clients=ATTACKER,
                                            clip_norm=DP_CLIP)
    sc["attack_scale_x1_fedmedian"] = _fl(aggregation="fedmedian", attack="scale", malicious_clients=ATTACKER)
    for name, ov in ADAPTIVE.items():
        for a in AGGS:
            sc[f"adaptive_{name}_x2_{a}"] = _fl(aggregation=a, malicious_clients=TWO_ATTACKERS, **ov)
    return sc


_DATA: dict = {}


def load_split(seed: int):
    """(client_data, global_test, n_features) for one seed; the CSV is read once per process."""
    from sklearn.preprocessing import StandardScaler
    from src.experiments.real_data_study import load, make_banks

    if "xy" not in _DATA:
        X, y = load()
        cut = int(len(X) * 0.8)
        sc = StandardScaler().fit(X[:cut])
        _DATA["xy"] = (sc.transform(X[:cut]).astype(np.float32), y[:cut],
                       sc.transform(X[cut:]).astype(np.float32), y[cut:])
    Xtr, ytr, Xte, yte = _DATA["xy"]
    banks = make_banks(Xtr, ytr, np.random.default_rng(seed))
    client_data = {b: {"train": banks[i], "test": (Xte, yte)} for i, b in enumerate(BANK_IDS)}
    return client_data, (Xte, yte), Xtr.shape[1]


def run_one(name: str, spec: dict, seed: int, data=None) -> dict:
    from src.experiments.framework import run_fl_inprocess, seed_everything
    from src.experiments.runners import run_centralized, run_local

    cfg = replace(load_config(), seed=seed, rounds=ROUNDS)
    for k, v in spec["overrides"].items():
        setattr(cfg, k, v)
    seed_everything(seed)
    client_data, global_test, n_features = data or load_split(seed)
    t0 = time.time()
    extra = {}
    if spec["kind"] == "central":
        res = run_centralized(cfg, client_data, global_test, n_features)
        hist, per_client = res["history_centralized"], res["per_client_history"]
    elif spec["kind"] == "local":
        res = run_local(cfg, client_data, global_test, n_features)
        hist, per_client = res["history_centralized"], res["per_client_history"]
    else:
        cfg.per_client_eval = False  # every bank's test set is the shared test split
        hist, strat, _ = run_fl_inprocess(cfg, client_data, global_test, n_features=n_features)
        per_client = {}
        extra = _phase_e_fields(cfg, strat)
        norms = getattr(strat, "update_norm_history", {})
        if norms:
            extra["update_norms"] = norms
    row = {
        "scenario": name, "seed": seed, "kind": spec["kind"], "overrides": spec["overrides"],
        "global_final": {k: v[-1][1] for k, v in hist.items()},
        "pr_auc_by_round": [v for _, v in hist.get("pr_auc", [])],
        "per_bank_final": per_client.get(f"round_{cfg.rounds}", {}),
        "n_test": int(len(global_test[1])), "n_test_pos": int(global_test[1].sum()),
        "seconds": round(time.time() - t0, 1),
    }
    row.update(extra)
    return row


def _run_job(job):
    import torch
    torch.set_num_threads(1)
    return run_one(*job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--only", default=None, help="regex on scenario name")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="parallel runs (each single-threaded)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs_path = OUT_DIR / "runs.jsonl"
    if not args.report_only:
        done = {(r["scenario"], r["seed"]) for r in load_runs(runs_path)}
        todo = [(n, s, seed) for n, s in build_scenarios().items()
                if not args.only or re.search(args.only, n) for seed in args.seeds if (n, seed) not in done]
        print(f"{len(todo)} runs to do ({len(done)} already done)", flush=True)
        if args.workers > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(args.workers) as pool:
                _write_rows(pool.imap_unordered(_run_job, todo), runs_path, len(todo))
        else:
            _write_rows((run_one(*job) for job in todo), runs_path, len(todo))
    from src.experiments.real_data_e_report import write_report
    write_report(runs_path, MD_PATH)


def _write_rows(rows, runs_path, total):
    for i, row in enumerate(rows, 1):
        with open(runs_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{i}/{total}] {row['scenario']} seed={row['seed']} "
              f"pr_auc={row['global_final'].get('pr_auc'):.4f} ({row['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
