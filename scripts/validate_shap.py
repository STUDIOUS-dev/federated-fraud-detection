import numpy as np
import torch
from src.config import load_config
from src.experiments.framework import ExperimentFramework
from src.model import FraudMLP
from src.explainability import get_explainer, explain_transaction, generate_human_explanation
from src.feature_engineering import BASE_FEATURES

def validate_shap():
    cfg = load_config()
    framework = ExperimentFramework("exp_shap", cfg)
    _, (global_test_X, global_test_y) = framework.prepare_data()
    
    n_features = framework.get_n_features()
    model = FraudMLP(n_features, cfg.hidden_dim, cfg.dropout)
    model.eval()
    
    # Pick 2 transactions (one fraud, one non-fraud if possible)
    fraud_indices = np.where(global_test_y == 1)[0]
    non_fraud_indices = np.where(global_test_y == 0)[0]
    
    if len(fraud_indices) > 0 and len(non_fraud_indices) > 0:
        idx1 = fraud_indices[0]
        idx2 = non_fraud_indices[0]
        indices = [idx1, idx2]
    else:
        indices = [0, 1]
        
    X_sample = global_test_X[indices]
    y_sample = global_test_y[indices]
    
    # Predict
    with torch.no_grad():
        X_tensor = torch.tensor(X_sample, dtype=torch.float32)
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).numpy().flatten()
        preds = (probs > 0.5).astype(int)
        
    # SHAP
    bg_data = global_test_X[:100]
    explainer = get_explainer(model, bg_data)
    
    for i, idx in enumerate(indices):
        print(f"\n--- Transaction {i+1} (True Label: {y_sample[i]}) ---")
        print(f"Prediction: {preds[i]}, Probability: {probs[i]:.4f}")
        
        instance_X = X_sample[i:i+1] # Keep 2D
        shap_values = explain_transaction(explainer, instance_X)
        
        human_exp = generate_human_explanation(shap_values, BASE_FEATURES)
        print(human_exp)
        
if __name__ == "__main__":
    validate_shap()
