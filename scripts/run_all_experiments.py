"""Regenerate experiments 1-10 (Flower + Ray, ~20 min).

DESTRUCTIVE: deletes results/summary.jsonl and overwrites every frozen results/exp*/ output that the
dashboard and README depend on. Requires `pip install "flwr[simulation]"` and an explicit --yes.
"""
import argparse
import json
import os
import subprocess
import sys
import time

EXPS = [
    "src.experiments.exp1_centralized",
    "src.experiments.exp2_local",
    "src.experiments.exp3_fedavg",
    "src.experiments.exp4_fedprox",
    "src.experiments.exp5_iid",
    "src.experiments.exp6_sizes",
    "src.experiments.exp7_imbalance",
    "src.experiments.exp8_dropout",
    "src.experiments.exp9_malicious",
    "src.experiments.exp10_defense",
]
SUMMARY_FILE = "results/summary.jsonl"


def _last_two_exp5():
    recs = [json.loads(l) for l in open(SUMMARY_FILE) if l.strip()]
    return [r["global_metrics"]["pr_auc"] for r in recs if r["exp_name"] == "exp5_iid"][-2:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true",
                    help="confirm that results/summary.jsonl may be deleted and all frozen results overwritten")
    args = ap.parse_args()
    if not args.yes:
        print("REFUSING to run: this deletes results/summary.jsonl and overwrites every frozen result "
              "under results/. Re-run with --yes to confirm.", file=sys.stderr)
        return 1

    if os.path.exists(SUMMARY_FILE):
        os.remove(SUMMARY_FILE)
        print("Cleared existing summary.jsonl")

    print("Running all experiments sequentially...")
    for exp in EXPS:
        print("\n======================================")
        print(f"Starting {exp}")
        print("======================================")
        t0 = time.time()
        subprocess.run([sys.executable, "-m", exp], check=True)
        print(f"{exp} finished in {time.time() - t0:.2f}s")

    # Reproducibility check: training is fully seeded, so a repeat run must give an identical PR-AUC.
    print("\n======================================")
    print("Running Reproducibility Check (exp5_iid, second run)")
    print("======================================")
    subprocess.run([sys.executable, "-m", "src.experiments.exp5_iid"], check=True)
    first, second = _last_two_exp5()
    print(f"exp5_iid PR-AUC run1={first!r} run2={second!r}")
    assert first == second, "Runs are not reproducible"
    print("\nAll experiments completed. For the multi-seed study run: python -m src.experiments.study")
    return 0


if __name__ == "__main__":
    sys.exit(main())
