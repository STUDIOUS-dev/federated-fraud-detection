from src.experiments.framework import run_fl_experiment

if __name__ == "__main__":
    print("Running Experiment 7: Extreme Class Imbalance...")
    # By using a tiny client, we almost guarantee 0 fraud cases in their split, testing evaluation robustness.
    run_fl_experiment("exp7_imbalance", {"client_sizes": [10, 500, 500, 500]})
