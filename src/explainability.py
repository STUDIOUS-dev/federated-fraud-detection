import shap
import torch
import numpy as np

class SHAPWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x).unsqueeze(-1)

def get_explainer(model: torch.nn.Module, background_X: np.ndarray):
    """Initializes the SHAP Explainer."""
    model.eval()
    wrapped_model = SHAPWrapper(model)
    background_tensor = torch.from_numpy(background_X).float()
    
    try:
        explainer = shap.DeepExplainer(wrapped_model, background_tensor)
    except Exception as e:
        print(f"DeepExplainer failed ({e}), falling back to GradientExplainer")
        try:
            explainer = shap.GradientExplainer(wrapped_model, background_tensor)
        except Exception as e2:
            print(f"GradientExplainer failed ({e2}), falling back to KernelExplainer")
            # KernelExplainer expects a function that returns numpy arrays
            def predict_fn(x):
                with torch.no_grad():
                    return model(torch.from_numpy(x).float()).numpy()
            # Kernel explainer can be slow, use KMeans to summarize background if too large
            background_summary = shap.kmeans(background_X, 50) if len(background_X) > 50 else background_X
            explainer = shap.KernelExplainer(predict_fn, background_summary)
            
    return explainer

def explain_transaction(explainer, instance_X: np.ndarray) -> np.ndarray:
    """Computes SHAP values for a single transaction instance."""
    # For PyTorch explainers we need tensors, for Kernel we need numpy
    if isinstance(explainer, shap.KernelExplainer):
        shap_values = explainer.shap_values(instance_X)
    else:
        instance_tensor = torch.from_numpy(instance_X).float()
        shap_values = explainer.shap_values(instance_tensor)
        
    # Standardize output to 1D numpy array
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    # Handle SHAP versions that wrap outputs differently
    if hasattr(shap_values, 'values'):
        # For newer shap.Explainer API
        shap_values = shap_values.values
        
    return np.array(shap_values).flatten()

def generate_human_explanation(shap_values: np.ndarray, feature_names: list[str], top_k: int = 4) -> str:
    """Generates a human-readable explanation based on top SHAP values."""
    if np.all(shap_values == 0):
        return "No specific factors strongly influenced this decision."
        
    top_indices = np.argsort(np.abs(shap_values))[-top_k:][::-1]
    
    factors = []
    for idx in top_indices:
        val = shap_values[idx]
        feat = feature_names[idx].replace("_", " ")
        if val == 0:
            continue
        direction = "increased" if val > 0 else "decreased"
        factors.append(f"{feat} ({direction} fraud probability)")
        
    if not factors:
        return "No significant factors found."
        
    return "Main factors:\n- " + "\n- ".join(factors)
