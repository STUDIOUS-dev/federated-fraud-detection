from src.experiments.framework import run_fl_experiment

if __name__ == "__main__":
    print("Running Experiment 6: Unequal Client Sizes...")
    # Varying from 50 (tiny) to 1000 (huge)
    run_fl_experiment("exp6_sizes", {"client_sizes": [200, 50, 1000, 750]})
