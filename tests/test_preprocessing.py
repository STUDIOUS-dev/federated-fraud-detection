import pandas as pd
import numpy as np
import pytest
from src.preprocessing import temporal_split, fit_scaler, to_xy
from src.feature_engineering import BASE_FEATURES

def test_temporal_split():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=10, freq="h"),
        "val": range(10)
    })
    train, test = temporal_split(df, test_fraction=0.2)
    
    assert len(train) == 8
    assert len(test) == 2
    assert train["timestamp"].max() < test["timestamp"].min()

def test_fit_scaler_and_to_xy():
    df1 = pd.DataFrame({
        "is_fraud": [0, 1],
    })
    for f in BASE_FEATURES:
        df1[f] = [10.0, 20.0]
        
    df2 = pd.DataFrame({
        "is_fraud": [0, 0],
    })
    for f in BASE_FEATURES:
        df2[f] = [15.0, 25.0]

    scaler = fit_scaler([df1, df2])
    
    # Check transformed shape and type
    X, y = to_xy(df1, scaler)
    assert X.shape == (2, len(BASE_FEATURES))
    assert X.dtype == np.float32
    assert y.shape == (2,)
    assert y.dtype == np.float32
    assert list(y) == [0.0, 1.0]
