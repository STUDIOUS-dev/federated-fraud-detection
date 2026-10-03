import joblib
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from src.model import FraudMLP
from src.feature_engineering import BASE_FEATURES, add_causal_features
from src.preprocessing import to_xy, temporal_split
from src.config import load_config
from src.data_generation import generate_federation

class ModelService:
    def __init__(self, exp_name: str = "exp10_defense", base_dir: str = "results"):
        self.exp_name = exp_name
        self.base_dir = Path(base_dir)
        self.exp_dir = self.base_dir / exp_name
        
        self.model = None
        self.scaler = None
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
            import yaml
            raw = yaml.safe_load(f)
        from src.config import Config
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
            
        if not hasattr(self, 'cfg'):
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
            
        # Check required columns (from BASE_FEATURES and 'is_fraud' for to_xy)
        missing = [f for f in BASE_FEATURES if f not in raw_tx_df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
            
        # Temporarily add is_fraud if missing to reuse to_xy
        added_dummy = False
        if "is_fraud" not in raw_tx_df.columns:
            raw_tx_df["is_fraud"] = 0
            added_dummy = True
            
        X, _ = to_xy(raw_tx_df, self.scaler)
        
        if added_dummy:
            raw_tx_df.drop(columns=["is_fraud"], inplace=True)
            
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
        if not hasattr(self, 'cfg'):
            self.load_model()
            
        return {
            "n_features": len(BASE_FEATURES),
            "expected_features": BASE_FEATURES.copy(),
            "hidden_dim": self.cfg.hidden_dim,
            "dropout": self.cfg.dropout
        }
