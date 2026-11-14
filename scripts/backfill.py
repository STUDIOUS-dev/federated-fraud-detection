"""LEGACY / DANGEROUS: do not run.

Appends records to results/summary.jsonl built from hard-coded guesses (local_epochs=1, malicious
client index 0, aggregation guessed from the experiment name, ...) that do NOT match how the frozen
experiments were actually run. The dashboard reads the LAST record per experiment, so one accidental
run silently corrupts what it shows. The script therefore refuses to run unless --force is given.
"""
import argparse
import json
import glob
import sys
from pathlib import Path


def backfill():
    summary_path = Path("results/summary.jsonl")
    for f in sorted(glob.glob("results/exp*/results.json")):
        with open(f, "r") as fp:
            data = json.load(fp)

        cfg_overrides = data.get("config", {})
        history = data.get("history_centralized", {})
        per_client = data.get("per_client_history", {})

        # We don't have the full original config, but we can reconstruct what we need for summary
        exp_name = data.get("exp_name", "")
        # Defaults
        aggregation = "FedAvg"
        if "fedprox" in exp_name.lower() or cfg_overrides.get("aggregation") == "fedprox":
            aggregation = "fedprox"

        malicious_idx = cfg_overrides.get("malicious_client_index", -1)
        if "malicious" in exp_name.lower():
            malicious_idx = 0

        summary = {
            "exp_name": exp_name,
            "aggregation": cfg_overrides.get("aggregation", aggregation),
            "seed": 42,
            "clients": 4,
            "rounds": 5,
            "local_epochs": 1,
            "lr": 0.001,
            "fedprox_mu": 0.01 if aggregation == "fedprox" else None,
            "malicious_client_index": malicious_idx,
            "client_sizes": cfg_overrides.get("client_sizes", [500, 500, 500, 500]),
            "participating_clients": data.get("participating_clients", {}),
            "global_metrics": {
                metric: values[-1][1] if values else None
                for metric, values in history.items()
            },
            "per_client_metrics": {
                k: v for k, v in per_client.get(f"round_{5}", {}).items()
            },
            "execution_time": 0.0,
            "status": "success"
        }
        with open(summary_path, "a") as fp:
            fp.write(json.dumps(summary) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Legacy summary.jsonl backfill (dangerous).")
    ap.add_argument("--force", action="store_true",
                    help="really append wrong, hard-coded records to results/summary.jsonl")
    args = ap.parse_args()
    if not args.force:
        print("REFUSING to run: backfill.py appends records with hard-coded WRONG values to "
              "results/summary.jsonl, and the dashboard uses the last record per experiment, so this "
              "would corrupt the frozen results. Re-run with --force only if you really mean it.",
              file=sys.stderr)
        return 1
    backfill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
