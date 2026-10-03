import pytest
import numpy as np
import torch
from src.model import FraudMLP
from src.explainability import get_explainer, explain_transaction, generate_human_explanation
from src.feature_engineering import BASE_FEATURES

def test_explainability_pipeline():
    n_features = len(BASE_FEATURES)
    model = FraudMLP(n_features)
    
    # Generate mock data
    background_X = np.random.randn(10, n_features).astype(np.float32)
    instance_X = np.random.randn(1, n_features).astype(np.float32)
    
    # Test explainer initialization
    explainer = get_explainer(model, background_X)
    assert explainer is not None
    
    # Test single transaction explanation
    shap_values = explain_transaction(explainer, instance_X)
    assert shap_values.shape == (n_features,)
    
    # Test human-readable generation
    # Force a specific shap array to test formatting
    mock_shap = np.zeros(n_features)
    mock_shap[0] = 0.5  # Positive impact
    mock_shap[1] = -0.3 # Negative impact
    
    explanation = generate_human_explanation(mock_shap, BASE_FEATURES, top_k=2)
    
    assert "Main factors" in explanation
    assert BASE_FEATURES[0].replace("_", " ") in explanation
    assert "increased" in explanation
    assert BASE_FEATURES[1].replace("_", " ") in explanation
    assert "decreased" in explanation
