import pytest
import pandas as pd
import numpy as np
from src.services.model import ModelService

def test_model_service_load_and_predict():
    # Use the real frozen artifacts from exp10_defense
    svc = ModelService(exp_name="exp10_defense", base_dir="results")
    svc.load_model()
    
    assert svc.model is not None
    assert svc.scaler is not None
    
    # Test transaction selection
    df = svc.get_test_transactions()
    assert len(df) > 0
    
    tx = svc.get_test_transaction(0)
    assert len(tx) == 1
    
    # Predict output shape and probability range
    prob = svc.predict_proba(tx)
    assert prob.shape == (1,)
    assert 0.0 <= prob[0] <= 1.0
    
    pred = svc.predict(tx)
    assert pred.shape == (1,)
    assert pred[0] in [0, 1]

def test_model_service_invalid_input():
    svc = ModelService(exp_name="exp10_defense", base_dir="results")
    svc.load_model()
    
    bad_df = pd.DataFrame({"random_col": [1.0]})
    with pytest.raises(ValueError, match="Missing required features"):
        svc.predict_proba(bad_df)

def test_preprocessing_parity():
    svc = ModelService(exp_name="exp10_defense", base_dir="results")
    svc.load_model()
    
    # Load regenerated raw data
    raw_df = svc.get_test_transactions()
    
    # Process it via the Model Service's preprocessor
    X_regen_tensor = svc._preprocess(raw_df)
    X_regen = X_regen_tensor.numpy()
    
    # Load the exact frozen artifacts
    frozen_X = np.load(svc.exp_dir / "test_sample_X.npy")
    
    # The frozen artifact may just be the first 100 samples
    subset_len = len(frozen_X)
    X_regen_subset = X_regen[:subset_len]
    
    # Verify exact shape and numeric parity
    assert X_regen_subset.shape == frozen_X.shape
    np.testing.assert_allclose(X_regen_subset, frozen_X, atol=1e-5)


def _svc():
    svc = ModelService(exp_name="exp10_defense", base_dir="results")
    svc.load_model()
    return svc


def test_preprocess_does_not_mutate_input():
    svc = _svc()
    df = svc.get_test_transactions().head(5).copy()
    cols, snapshot = list(df.columns), df.copy()
    svc._preprocess(df)
    svc.predict_proba(df)
    assert list(df.columns) == cols and "is_fraud" not in set(cols) - set(snapshot.columns)
    pd.testing.assert_frame_equal(df, snapshot)
    no_label = df[svc_features(svc)].copy()
    svc.predict_proba(no_label)
    assert "is_fraud" not in no_label.columns


def svc_features(svc):
    from src.feature_engineering import BASE_FEATURES
    return BASE_FEATURES


def test_preprocess_rejects_nan_inf_and_non_numeric():
    svc = _svc()
    base = svc.get_test_transactions()[svc_features(svc)].head(3).copy()
    col = svc_features(svc)[0]
    for bad in (np.nan, np.inf, -np.inf):
        df = base.copy(); df[col] = df[col].astype(float); df.iloc[0, df.columns.get_loc(col)] = bad
        with pytest.raises(ValueError, match="finite"):
            svc.predict_proba(df)
    df = base.copy().astype(object); df.iloc[0, df.columns.get_loc(col)] = "abc"
    with pytest.raises(ValueError, match="numeric"):
        svc.predict_proba(df)
