from src.experiments.framework import run_fl_experiment

if __name__ == "__main__":
    print("Running Experiment 8: Client Dropout...")
    # fraction_fit=0.5 out of 4 clients means 2 clients participate each round
    run_fl_experiment("exp8_dropout", {"fraction_fit": 0.5})
