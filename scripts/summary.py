"""Print final-round global and per-client PR-AUC for every results/exp*/results.json (read-only)."""
import glob
import json


def main() -> None:
    print("Summary of Experiments (Global PR-AUC at final round):")
    for f in sorted(glob.glob("results/exp*/results.json")):
        with open(f, "r") as fp:
            data = json.load(fp)
        try:
            global_pr_auc = data["history_centralized"]["pr_auc"][-1][1]
            rounds = data["per_client_history"]
            last = max(rounds, key=lambda k: int(k.split("_")[1]))
            per_client = {k: v["pr_auc"] for k, v in rounds[last].items()}
            print(f"{f}:")
            print(f"  Global PR-AUC: {global_pr_auc:.4f}")
            print(f"  Per-Client PR-AUC ({last.replace('_', ' ')}): {per_client}")
        except (KeyError, ValueError) as e:
            print(f"{f}: Could not parse results ({e})")


if __name__ == "__main__":
    main()
