import numpy as np
from src.config import load_config
from src.experiments.framework import ExperimentFramework
from src.model import FraudMLP, state_to_numpy, numpy_to_state, predict
from src.federated.local_training import train_local
from src.evaluation import evaluate_probs

def run_exp1_centralized():
    cfg = load_config()
    framework = ExperimentFramework("exp1_centralized", cfg)
    client_data, (global_test_X, global_test_y) = framework.prepare_data()
    
    # Pool training data
    train_X_all = np.vstack([data["train"][0] for data in client_data.values()])
    train_y_all = np.concatenate([data["train"][1] for data in client_data.values()])
    
    # Initialize Model
    n_features = framework.get_n_features()
    model = FraudMLP(n_features, cfg.hidden_dim, cfg.dropout)
    state = state_to_numpy(model)
    
    results = {
        "history_centralized": {},
        "per_client_history": {}
    }
    print(f"Training Centralized Model on {len(train_X_all)} samples...")
    # Train sequentially for all rounds
    for r in range(1, cfg.rounds + 1):
        state, _ = train_local(
            initial_state=state,
            X=train_X_all,
            y=train_y_all,
            n_features=n_features,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            lr=cfg.lr,
            local_epochs=cfg.local_epochs,
            batch_size=cfg.batch_size,
            prox_mu=0.0
        )
        
        numpy_to_state(model, state)
        
        # Evaluate Globally
        if cfg.global_eval:
            global_metrics = evaluate_probs(global_test_y, predict(model, global_test_X))
            print(f"Round {r} Global PR-AUC: {global_metrics['pr_auc']:.4f}")
            for k, v in global_metrics.items():
                if k not in results["history_centralized"]:
                    results["history_centralized"][k] = []
                results["history_centralized"][k].append((r, v))
        # Evaluate Per-Client
        if cfg.per_client_eval:
            client_metrics = {}
            for bank_id, data in client_data.items():
                metrics = evaluate_probs(data["test"][1], predict(model, data["test"][0]))
                client_metrics[bank_id] = metrics
            results["per_client_history"][f"round_{r}"] = client_metrics
    framework.save_results(results)
    print("Experiment 1 Centralized complete. Results saved.")

if __name__ == "__main__":
    run_exp1_centralized()
