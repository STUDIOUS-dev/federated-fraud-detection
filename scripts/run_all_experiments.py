import subprocess
import os
import time

exps = [
    "src.experiments.exp1_centralized",
    "src.experiments.exp2_local",
    "src.experiments.exp3_fedavg",
    "src.experiments.exp4_fedprox",
    "src.experiments.exp5_iid",
    "src.experiments.exp6_sizes",
    "src.experiments.exp7_imbalance",
    "src.experiments.exp8_dropout",
    "src.experiments.exp9_malicious",
    "src.experiments.exp10_defense"
]

summary_file = "results/summary.jsonl"
if os.path.exists(summary_file):
    os.remove(summary_file)
    print("Cleared existing summary.jsonl")

print("Running all experiments sequentially...")
for exp in exps:
    print(f"\n======================================")
    print(f"Starting {exp}")
    print(f"======================================")
    t0 = time.time()
    subprocess.run(["python", "-m", exp], check=True)
    print(f"{exp} finished in {time.time() - t0:.2f}s")

# Reproducibility check: training is fully seeded, so a repeat run must give an identical PR-AUC.
print("\n======================================")
print("Running Reproducibility Check (exp5_iid, second run)")
print("======================================")
import json
def last_two_exp5():
    recs = [json.loads(l) for l in open(summary_file) if l.strip()]
    return [r["global_metrics"]["pr_auc"] for r in recs if r["exp_name"] == "exp5_iid"][-2:]
subprocess.run(["python", "-m", "src.experiments.exp5_iid"], check=True)
first, second = last_two_exp5()
print(f"exp5_iid PR-AUC run1={first!r} run2={second!r}")
assert first == second, "Runs are not reproducible"
print("\nAll experiments completed. For the multi-seed study run: python -m src.experiments.study")
