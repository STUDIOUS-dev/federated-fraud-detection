from src.experiments.framework import run_fl_experiment

if __name__ == "__main__":
    print("Running Experiment 9: Malicious Client Poisoning...")
    # Client 3 (bank_d) acts maliciously to corrupt the global model
    run_fl_experiment("exp9_malicious", {"malicious_client_index": 3})
