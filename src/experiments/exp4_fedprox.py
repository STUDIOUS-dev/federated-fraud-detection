from src.experiments.framework import run_fl_experiment

def run_exp4_fedprox():
    run_fl_experiment("exp4_fedprox", {"aggregation": "fedprox", "fedprox_mu": 0.1})
    
if __name__ == "__main__":
    run_exp4_fedprox()
