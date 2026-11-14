from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from src.config import Config
from src.data_generation import generate_federation
from src.feature_engineering import BASE_FEATURES, add_causal_features
from src.model import FraudMLP
from src.preprocessing import temporal_split

class ModelService:
    def __init__(self, exp_name: str = "exp10_defense", base_dir: str = "results"):
        self.exp_name = exp_name
        self.base_dir = Path(base_dir)
        self.exp_dir = self.base_dir / exp_name
        
        self.model = None
        self.scaler = None
        self.cfg = None
        self.test_data_cache = None
        
    def load_model(self):
        """Loads the pre-trained model and preprocessing scaler."""
        model_path = self.exp_dir / "model.pt"
        scaler_path = self.exp_dir / "scaler.pkl"
        config_path = self.exp_dir / "config.yaml"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")
            
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
        self.cfg = Config(**raw)
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)
        
        # Initialize and load model
        self.model = FraudMLP(
            n_features=len(BASE_FEATURES), 
            hidden_dim=self.cfg.hidden_dim, 
            dropout=self.cfg.dropout
        )
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    def get_test_transactions(self) -> pd.DataFrame:
        """Deterministically rebuilds the raw test transactions for the experiment."""
        if self.test_data_cache is not None:
            return self.test_data_cache
            
        if self.cfg is None:
            self.load_model()
            
        # Recreate data using exact configuration and seeds
        banks = generate_federation(self.cfg)
        banks = {k: add_causal_features(v) for k, v in banks.items()}
        splits = {k: temporal_split(v, self.cfg.test_fraction) for k, v in banks.items()}
        
        # Extract and combine all test chunks
        test_frames = [te for _, te in splits.values()]
        self.test_data_cache = pd.concat(test_frames).reset_index(drop=True)
        return self.test_data_cache

    def get_test_transaction(self, index: int) -> pd.DataFrame:
        """Selects a specific transaction from the test set by index."""
        df = self.get_test_transactions()
        if index < 0 or index >= len(df):
            raise IndexError("Transaction index out of bounds")
        return df.iloc[[index]].copy()

    def _preprocess(self, raw_tx_df: pd.DataFrame) -> torch.Tensor:
        """Applies exact feature ordering and scaling to raw transactions."""
        if self.scaler is None:
            self.load_model()
            
        missing = [f for f in BASE_FEATURES if f not in raw_tx_df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        feats = raw_tx_df[BASE_FEATURES]  # new frame; caller's frame is untouched
        try:
            feats = feats.astype(float)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Feature values must be numeric: {e}") from e
        values = feats.to_numpy()
        if not np.isfinite(values).all():
            bad = [f for f in BASE_FEATURES if not np.isfinite(feats[f].to_numpy()).all()]
            raise ValueError(f"Feature values must be finite (no NaN/inf); offending features: {bad}")

        X = self.scaler.transform(feats).astype(np.float32)
        return torch.from_numpy(X).float()

    def predict_proba(self, raw_tx_df: pd.DataFrame) -> np.ndarray:
        """Returns fraud probability for raw transactions."""
        X_tensor = self._preprocess(raw_tx_df)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).numpy()
        return probs
        
    def predict(self, raw_tx_df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Returns binary predictions for raw transactions."""
        probs = self.predict_proba(raw_tx_df)
        return (probs >= threshold).astype(int)

    def get_model_metadata(self) -> dict:
        """Returns metadata about the loaded model and its expectations."""
        if self.cfg is None:
            self.load_model()
            
        return {
            "n_features": len(BASE_FEATURES),
            "expected_features": BASE_FEATURES.copy(),
            "hidden_dim": self.cfg.hidden_dim,
            "dropout": self.cfg.dropout
        }
