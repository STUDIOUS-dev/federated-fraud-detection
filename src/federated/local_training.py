from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.model import FraudMLP, numpy_to_state, state_to_numpy


def class_weight(y: np.ndarray) -> float:
    positives = max(float(y.sum()), 1.0)
    negatives = max(float(len(y) - y.sum()), 1.0)
    return negatives / positives


def train_local(
    initial_state: list[np.ndarray], X: np.ndarray, y: np.ndarray,
    n_features: int, hidden_dim: int, dropout: float, lr: float,
    local_epochs: int, batch_size: int, prox_mu: float = 0.0, seed: int | None = None,
) -> tuple[list[np.ndarray], int]:
    gen = None
    if seed is not None:
        torch.manual_seed(seed)  # dropout masks
        gen = torch.Generator().manual_seed(seed)  # shuffle order
    model = FraudMLP(n_features, hidden_dim, dropout)
    numpy_to_state(model, initial_state)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor([class_weight(y)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    global_params = [p.detach().clone() for p in model.parameters()]

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)
    for _ in range(local_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            if prox_mu > 0:
                prox = sum(torch.sum((p - gp) ** 2) for p, gp in zip(model.parameters(), global_params))
                loss = loss + (prox_mu / 2.0) * prox
            loss.backward()
            optimizer.step()
    return state_to_numpy(model), len(y)


def fedavg(client_states: list[list[np.ndarray]], client_sizes: list[int]) -> list[np.ndarray]:
    total = float(sum(client_sizes))
    out = []
    for layer in zip(*client_states):
        agg = sum(layer_i * (size / total) for layer_i, size in zip(layer, client_sizes))
        out.append(agg.astype(np.float32))
    return out
