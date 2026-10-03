from src.experiments.framework import run_fl_experiment

if __name__ == "__main__":
    print("Running Experiment 5: IID configuration...")
    run_fl_experiment("exp5_iid", {"heterogeneity": "iid"})
