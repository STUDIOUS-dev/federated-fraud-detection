import numpy as np
import torch
from src.services.model import ModelService
from src.feature_engineering import BASE_FEATURES

def verify_provenance():
    svc = ModelService("exp10_defense")
    svc.load_model()
    
    # Load regenerated test dataset
    raw_test_df = svc.get_test_transactions()
    
    # Preprocess it through the service
    X_regen_tensor = svc._preprocess(raw_test_df)
    X_regen = X_regen_tensor.numpy()
    
    # We must construct y_regen keeping the same order
    y_regen = raw_test_df["is_fraud"].values
    
    # Load frozen artifacts
    X_frozen = np.load(svc.exp_dir / "test_sample_X.npy")
    y_frozen = np.load(svc.exp_dir / "test_sample_y.npy")
    
    print(f"Regenerated shape: X={X_regen.shape}, y={y_regen.shape}")
    print(f"Frozen shape:      X={X_frozen.shape}, y={y_frozen.shape}")
    
    # The frozen one might be just a subset (first 100 samples in framework.py)
    # Check if they match
    n_frozen = len(X_frozen)
    
    # Compare feature count
    assert X_regen.shape[1] == X_frozen.shape[1], "Feature count mismatch"
    assert X_regen.shape[1] == len(BASE_FEATURES), "Feature count != BASE_FEATURES"
    
    # Compare first N samples numerically
    X_regen_subset = X_regen[:n_frozen]
    y_regen_subset = y_regen[:n_frozen]
    
    try:
        np.testing.assert_allclose(X_regen_subset, X_frozen, atol=1e-5)
        print("X values match within tolerance.")
    except AssertionError as e:
        print("X values DO NOT match:", e)
        
    try:
        np.testing.assert_array_equal(y_regen_subset, y_frozen)
        print("y labels align perfectly.")
    except AssertionError as e:
        print("y labels DO NOT match:", e)

if __name__ == "__main__":
    verify_provenance()
