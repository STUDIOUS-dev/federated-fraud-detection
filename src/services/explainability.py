import numpy as np
import pandas as pd
import torch

from src.explainability import get_explainer, explain_transaction, generate_human_explanation
from src.services.model import ModelService

class ExplainabilityService:
    def __init__(self, model_service: ModelService):
        if model_service.model is None:
            model_service.load_model()
        self.model_service = model_service
        self.explainer = None
        self.expected_value = None

    def _initialize_explainer(self):
        """Initializes SHAP using a small background sample from the test set."""
        # Use first 100 test samples as background
        background_df = self.model_service.get_test_transactions().head(100)
        background_tensor = self.model_service._preprocess(background_df).numpy()
        
        self.explainer = get_explainer(self.model_service.model, background_tensor)
        
        # Cache expected value if available (for probability alignment checks)
        if hasattr(self.explainer, 'expected_value'):
            self.expected_value = self.explainer.expected_value

    def explain(self, raw_tx_df: pd.DataFrame) -> dict:
        """
        Computes SHAP values and a human explanation for a raw transaction dataframe.
        Verifies alignment with the model.
        """
        if len(raw_tx_df) != 1:
            raise ValueError("ExplainabilityService currently supports explaining one transaction at a time.")
            
        if self.explainer is None:
            self._initialize_explainer()
            
        metadata = self.model_service.get_model_metadata()
        expected_features = metadata["expected_features"]
        
        # Feature name alignment verification
        missing = [f for f in expected_features if f not in raw_tx_df.columns]
        if missing:
            raise ValueError(f"Transaction missing required features: {missing}")
            
        instance_tensor = self.model_service._preprocess(raw_tx_df)
        shap_values = explain_transaction(self.explainer, instance_tensor.numpy())
        
        # Dimension verification
        if len(shap_values) != metadata["n_features"]:
            raise ValueError(
                f"SHAP output dimension {len(shap_values)} does not match "
                f"model expected features {metadata['n_features']}"
            )
            
        # Probability alignment (optional check, but good for verification)
        model_prob = self.model_service.predict_proba(raw_tx_df)[0]
        
        # Note: We do not claim causal explanations here, only model attribution.
        explanation_text = generate_human_explanation(shap_values, expected_features)
        
        # Package the result
        return {
            "shap_values": shap_values.tolist(),
            "feature_names": expected_features,
            "explanation": explanation_text,
            "model_probability": float(model_prob),
            "base_value": float(self.expected_value) if self.expected_value is not None else None
        }
