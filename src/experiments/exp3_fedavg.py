from src.experiments.framework import run_fl_experiment

def run_exp3_fedavg():
    run_fl_experiment("exp3_fedavg", {"aggregation": "fedavg"})
    
if __name__ == "__main__":
    run_exp3_fedavg()
