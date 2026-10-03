"""
Experiment 10: Robust Aggregation Defense
Tests the effectiveness of FedMedian in defending against a malicious client.
"""

from src.experiments.framework import run_fl_experiment

def main():
    print("Running Experiment 10: Malicious Client Defense (FedMedian)...")
    config_overrides = {
        "aggregation": "FedMedian",
        "malicious_client_index": 0  # Same attacker as Exp 9
    }
    run_fl_experiment("exp10_defense", config_overrides)

if __name__ == "__main__":
    main()
