"""Phase E multi-seed study: communication cost, update clipping, client-level differential
privacy, and adaptive (omniscient) poisoning attacks.

Separate from the frozen v1.1 study: writes results/study_e/runs.jsonl, results/study_e/aggregate.json
and docs/PHASE_E_RESULTS.md. Resumable, like the v1.1 study. Baselines (central, local, FedAvg) are
re-run here so that every comparison is paired within one software environment.

    python -m src.experiments.study_e                    # all scenarios, seeds 42..46
    python -m src.experiments.study_e --only dp_ --seeds 42
    python -m src.experiments.study_e --report-only
    python -m src.experiments.study_e --workers 4         # parallel; rows are identical
"""
import argparse
import json
import re
from pathlib import Path

from src.experiments.study import SEEDS, load_runs, run_one

OUT_DIR = Path("results/study_e")
ATTACKER = [3]            # bank_d, as in the v1.1 study
TWO_ATTACKERS = [2, 3]    # 50% of the federation
AGGS = ["fedavg", "fedmedian", "fedtrimmedavg", "krum"]
DP_CLIP = 0.5             # ~ half the typical honest update norm (0.6-1.2 in round 1)
DP_NOISE = (0.5, 1.0, 2.0, 4.0)
ROUNDS = (1, 2, 3, 10)    # 5 rounds is the plain "fedavg" scenario
ADAPTIVE = {"alie": {"attack": "alie", "alie_z": 1.0},
            "alie_z3": {"attack": "alie", "alie_z": 3.0},
            "minmax": {"attack": "minmax"}}


def _fl(**ov):
    return {"kind": "fl", "overrides": ov}


def build_scenarios() -> dict:
    sc = {
        "central": {"kind": "central", "overrides": {}},
        "local": {"kind": "local", "overrides": {}},
        "fedavg": _fl(aggregation="fedavg"),
    }
    # E1 communication cost vs accuracy
    for r in ROUNDS:
        sc[f"fedavg_rounds{r}"] = _fl(aggregation="fedavg", rounds=r)
    # E2 clipping alone, then client-level DP at a fixed clip norm
    for c in (0.25, 0.5, 1.0):
        sc[f"fedavg_clip{c}"] = _fl(aggregation="fedavg", clip_norm=c)
    for z in DP_NOISE:
        sc[f"dp_z{z}"] = _fl(aggregation="fedavg", clip_norm=DP_CLIP, dp_noise_multiplier=z)
    sc["dp_z1.0_rounds10"] = _fl(aggregation="fedavg", clip_norm=DP_CLIP, dp_noise_multiplier=1.0, rounds=10)
    # Clipping / DP as a defence against the v1.1 scaling attacker
    sc["attack_scale_x1_fedavg"] = _fl(aggregation="fedavg", attack="scale", malicious_clients=ATTACKER)
    sc["attack_scale_x1_fedavg_clip"] = _fl(aggregation="fedavg", attack="scale", malicious_clients=ATTACKER,
                                            clip_norm=DP_CLIP)
    sc["attack_scale_x1_dp_z1.0"] = _fl(aggregation="fedavg", attack="scale", malicious_clients=ATTACKER,
                                        clip_norm=DP_CLIP, dp_noise_multiplier=1.0)
    # E4 adaptive attacks x aggregators, 1 and 2 attackers
    for name, ov in ADAPTIVE.items():
        for k, who in (("x1", ATTACKER), ("x2", TWO_ATTACKERS)):
            for a in AGGS:
                sc[f"adaptive_{name}_{k}_{a}"] = _fl(aggregation=a, malicious_clients=who, **ov)
    # Norm clipping in front of robust aggregation against the strongest adaptive setting
    for a in ("fedavg", "fedmedian", "krum"):
        sc[f"adaptive_alie_z3_x2_{a}_clip"] = _fl(aggregation=a, malicious_clients=TWO_ATTACKERS,
                                                  clip_norm=DP_CLIP, **ADAPTIVE["alie_z3"])
    return sc


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
                rows = pool.imap_unordered(_run_job, todo)
                _write_rows(rows, runs_path, len(todo))
        else:
            _write_rows((run_one(n, spec, seed) for n, spec, seed in todo), runs_path, len(todo))
    from src.experiments.study_e_report import write_report
    write_report(runs_path)


def _run_job(job):
    import torch
    torch.set_num_threads(1)  # one run per core; results are bit-identical to multi-threaded runs
    return run_one(*job)


def _write_rows(rows, runs_path, total):
    for i, row in enumerate(rows, 1):
        with open(runs_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{i}/{total}] {row['scenario']} seed={row['seed']} "
              f"pr_auc={row['global_final'].get('pr_auc'):.4f} ({row['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
