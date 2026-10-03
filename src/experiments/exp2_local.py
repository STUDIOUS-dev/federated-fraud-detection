import numpy as np
from src.config import load_config
from src.experiments.framework import ExperimentFramework
from src.model import FraudMLP, state_to_numpy, numpy_to_state, predict
from src.federated.local_training import train_local
from src.evaluation import evaluate_probs

def run_exp2_local():
    cfg = load_config()
    framework = ExperimentFramework("exp2_local", cfg)
    client_data, (global_test_X, global_test_y) = framework.prepare_data()
    n_features = framework.get_n_features()
    
    results = {}
    
    for bank_id, data in client_data.items():
        print(f"Training Local-Only Model for {bank_id} on {len(data['train'][0])} samples...")
        model = FraudMLP(n_features, cfg.hidden_dim, cfg.dropout)
        state = state_to_numpy(model)
        
        bank_results = {
            "history_centralized": {},
            "per_client_history": {}
        }
        
        # Train sequentially for all rounds
        for r in range(1, cfg.rounds + 1):
            state, _ = train_local(
                initial_state=state,
                X=data["train"][0],
                y=data["train"][1],
                n_features=n_features,
                hidden_dim=cfg.hidden_dim,
                dropout=cfg.dropout,
                lr=cfg.lr,
                local_epochs=cfg.local_epochs,
                batch_size=cfg.batch_size,
                prox_mu=0.0
            )
            numpy_to_state(model, state)
            
            if cfg.global_eval:
                global_metrics = evaluate_probs(global_test_y, predict(model, global_test_X))
                
            # Evaluate Local
            if cfg.per_client_eval:
                local_metrics = evaluate_probs(data["test"][1], predict(model, data["test"][0]))
                print(f"[{bank_id}] Round {r} Local PR-AUC: {local_metrics['pr_auc']:.4f}")
                
            if f"round_{r}" not in bank_results["per_client_history"]:
                bank_results["per_client_history"][f"round_{r}"] = {}
            if cfg.per_client_eval:
                bank_results["per_client_history"][f"round_{r}"][bank_id] = local_metrics
            
            if cfg.global_eval:
                for k, v in global_metrics.items():
                    if k not in bank_results["history_centralized"]:
                        bank_results["history_centralized"][k] = []
                    bank_results["history_centralized"][k].append((r, v))
                print(f"[{bank_id}] Round {r} Local PR-AUC: {local_metrics['pr_auc']:.4f}")
            
        results[bank_id] = bank_results
        
    # Aggregate local models' global metrics (average)
    agg_results = {
        "history_centralized": {},
        "per_client_history": {}
    }
    
    for bank_id, bank_res in results.items():
        # Per client
        for round_key, metrics in bank_res.get("per_client_history", {}).items():
            if round_key not in agg_results["per_client_history"]:
                agg_results["per_client_history"][round_key] = {}
            agg_results["per_client_history"][round_key].update(metrics)
            
        # Global metrics (we'll just take the average across all banks for simplicity)
        for metric_name, values in bank_res.get("history_centralized", {}).items():
            if metric_name not in agg_results["history_centralized"]:
                agg_results["history_centralized"][metric_name] = {}
            for r, val in values:
                if r not in agg_results["history_centralized"][metric_name]:
                    agg_results["history_centralized"][metric_name][r] = []
                agg_results["history_centralized"][metric_name][r].append(val)
                
    # Finalize global metrics by averaging
    for metric_name, round_vals in agg_results["history_centralized"].items():
        avg_vals = [(r, float(np.mean(vals))) for r, vals in round_vals.items()]
        agg_results["history_centralized"][metric_name] = avg_vals
            
    framework.save_results(agg_results)
    print("Experiment 2 Local-Only complete. Results saved.")

if __name__ == "__main__":
    run_exp2_local()
