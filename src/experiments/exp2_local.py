from src.config import load_config
from src.experiments.framework import ExperimentFramework
from src.experiments.runners import run_local


def run_exp2_local():
    cfg = load_config()
    framework = ExperimentFramework("exp2_local", cfg)
    client_data, global_test = framework.prepare_data()
    results = run_local(cfg, client_data, global_test, framework.get_n_features(), verbose=True)
    framework.save_results(results)
    print("Experiment 2 Local-Only complete. Results saved.")


if __name__ == "__main__":
    run_exp2_local()
