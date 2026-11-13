import pytest
from src.services.model import ModelService
from src.services.explainability import ExplainabilityService

def test_explainability_service_integration():
    model_svc = ModelService(exp_name="exp10_defense", base_dir="results")
    model_svc.load_model()
    
    expl_svc = ExplainabilityService(model_svc)
    
    tx = model_svc.get_test_transaction(0)
    
    result = expl_svc.explain(tx)
    
    assert "shap_values" in result
    assert "feature_names" in result
    assert "explanation" in result
    assert "model_probability" in result
    assert "base_value" in result
    # Regression: SHAP expected_value may be a 1-element array on newer NumPy
    assert type(result["base_value"]) is float

    metadata = model_svc.get_model_metadata()
    assert len(result["shap_values"]) == metadata["n_features"]
    assert len(result["feature_names"]) == metadata["n_features"]
    assert result["feature_names"] == metadata["expected_features"]
    
    # Must be between 0 and 1
    assert 0.0 <= result["model_probability"] <= 1.0
    
    # Explanation text should have something
    assert len(result["explanation"]) > 0

def test_explainability_service_multiple_transactions_fails():
    model_svc = ModelService(exp_name="exp10_defense", base_dir="results")
    model_svc.load_model()
    expl_svc = ExplainabilityService(model_svc)
    
    txs = model_svc.get_test_transactions().head(2)
    with pytest.raises(ValueError, match="supports explaining one transaction at a time"):
        expl_svc.explain(txs)
