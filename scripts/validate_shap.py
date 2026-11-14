"""Print SHAP explanations of the frozen model (results/exp10_defense/model.pt) for two test transactions.

Writes nothing: data comes from build_data() (no ExperimentFramework, so no config.yaml / scaler.pkl
are rewritten) and the checkpoint is only read.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from src.config import load_config
from src.experiments.framework import build_data
from src.model import FraudMLP
from src.explainability import get_explainer, explain_transaction, generate_human_explanation
from src.feature_engineering import BASE_FEATURES

EXP_DIR = "results/exp10_defense"


def validate_shap():
    cfg = load_config()
    _, (global_test_X, global_test_y), _ = build_data(cfg)

    n_features = len(BASE_FEATURES)
    model = FraudMLP(n_features, cfg.hidden_dim, cfg.dropout)
    model.load_state_dict(torch.load(f"{EXP_DIR}/model.pt", weights_only=True))
    model.eval()

    fraud_indices = np.where(global_test_y == 1)[0]
    non_fraud_indices = np.where(global_test_y == 0)[0]
    if len(fraud_indices) > 0 and len(non_fraud_indices) > 0:
        indices = [fraud_indices[0], non_fraud_indices[0]]
    else:
        indices = [0, 1]

    X_sample = global_test_X[indices]
    y_sample = global_test_y[indices]

    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X_sample, dtype=torch.float32))).numpy().flatten()
        preds = (probs > 0.5).astype(int)

    explainer = get_explainer(model, global_test_X[:100])

    for i, idx in enumerate(indices):
        print(f"\n--- Transaction {i+1} (True Label: {y_sample[i]}) ---")
        print(f"Prediction: {preds[i]}, Probability: {probs[i]:.4f}")
        shap_values = explain_transaction(explainer, X_sample[i:i+1])
        print(generate_human_explanation(shap_values, BASE_FEATURES))


if __name__ == "__main__":
    validate_shap()
