from src.experiments.framework import run_fl_experiment

if __name__ == "__main__":
    print("Running Experiment 8: Client Dropout...")
    # Each client independently fails to report in each round with p=0.5 (seeded; >=1 survivor).
    run_fl_experiment("exp8_dropout", {"dropout_rate": 0.5})
