import json, glob

print("Summary of Experiments (Global PR-AUC at final round):")
for f in sorted(glob.glob('results/exp*/results.json')):
    with open(f, 'r') as fp:
        data = json.load(fp)
        try:
            global_pr_auc = data['history_centralized']['pr_auc'][-1][1]
            per_client = {k: v['pr_auc'] for k, v in data['per_client_history']['round_5'].items()}
            print(f"{f}:")
            print(f"  Global PR-AUC: {global_pr_auc:.4f}")
            print(f"  Per-Client PR-AUC (Round 5): {per_client}")
        except KeyError as e:
            print(f"{f}: Could not parse results ({e})")
