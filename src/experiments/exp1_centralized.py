from src.config import load_config
from src.experiments.framework import ExperimentFramework
from src.experiments.runners import run_centralized


def run_exp1_centralized():
    cfg = load_config()
    framework = ExperimentFramework("exp1_centralized", cfg)
    client_data, global_test = framework.prepare_data()
    results = run_centralized(cfg, client_data, global_test, framework.get_n_features(), verbose=True)
    framework.save_results(results)
    print("Experiment 1 Centralized complete. Results saved.")


if __name__ == "__main__":
    run_exp1_centralized()
