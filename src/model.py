from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn

class FraudMLP(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def state_to_numpy(model: nn.Module) -> list[np.ndarray]:
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]


def numpy_to_state(model: nn.Module, arrays: list[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    state = {k: torch.tensor(a) for k, a in zip(keys, arrays)}
    model.load_state_dict(state)


def clone_state(arrays: list[np.ndarray]) -> list[np.ndarray]:
    return [a.copy() for a in arrays]

def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).numpy()
