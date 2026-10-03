from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .feature_engineering import BASE_FEATURES


def temporal_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = df.sort_values("timestamp").reset_index(drop=True)
    cut = int(len(x) * (1 - test_fraction))
    return x.iloc[:cut].copy(), x.iloc[cut:].copy()


def fit_scaler(train_frames: list[pd.DataFrame]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(pd.concat(train_frames)[BASE_FEATURES].astype(float))
    return scaler


def to_xy(df: pd.DataFrame, scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray]:
    X = scaler.transform(df[BASE_FEATURES].astype(float)).astype(np.float32)
    y = df["is_fraud"].to_numpy(dtype=np.float32)
    return X, y
