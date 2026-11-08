"""Non-federated reference runners (centralized and local-only). Pure functions: no disk writes.

Both return the same structure as Flower runs:
  {"history_centralized": {metric: [(round, value), ...]}, "per_client_history": {"round_k": {bank: metrics}}}
Training uses the same total rounds x local_epochs budget as the federated runs.
"""
import numpy as np
from src.evaluation import evaluate_probs
from src.federated.local_training import train_local
from src.model import FraudMLP, numpy_to_state, state_to_numpy, predict


def _init_state(cfg, n_features):
    import torch
    torch.manual_seed(cfg.seed)  # same initial weights as the federated runs
    return state_to_numpy(FraudMLP(n_features, cfg.hidden_dim, cfg.dropout))


def _train_rounds(cfg, X, y, n_features, seed_offset):
    state = _init_state(cfg, n_features)
    model = FraudMLP(n_features, cfg.hidden_dim, cfg.dropout)
    for r in range(1, cfg.rounds + 1):
        state, _ = train_local(
            initial_state=state, X=X, y=y, n_features=n_features, hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout, lr=cfg.lr, local_epochs=cfg.local_epochs,
            batch_size=cfg.batch_size, prox_mu=0.0, seed=cfg.seed * 100003 + r * 101 + seed_offset)
        numpy_to_state(model, state)
        yield r, model


def run_centralized(cfg, client_data, global_test, n_features, verbose=False):
    X = np.vstack([d["train"][0] for d in client_data.values()])
    y = np.concatenate([d["train"][1] for d in client_data.values()])
    res = {"history_centralized": {}, "per_client_history": {}}
    for r, model in _train_rounds(cfg, X, y, n_features, seed_offset=50):
        g = evaluate_probs(global_test[1], predict(model, global_test[0]))
        if verbose:
            print(f"Round {r} Global PR-AUC: {g['pr_auc']:.4f}")
        for k, v in g.items():
            res["history_centralized"].setdefault(k, []).append((r, v))
        res["per_client_history"][f"round_{r}"] = {
            b: evaluate_probs(d["test"][1], predict(model, d["test"][0])) for b, d in client_data.items()}
    return res


def run_local(cfg, client_data, global_test, n_features, verbose=False):
    """Each bank trains alone. 'Global' metrics = mean over banks of each local model on the global test pool."""
    per_bank_global = {}
    res = {"history_centralized": {}, "per_client_history": {}}
    for i, (bank, d) in enumerate(client_data.items()):
        for r, model in _train_rounds(cfg, d["train"][0], d["train"][1], n_features, seed_offset=i):
            per_bank_global.setdefault(r, []).append(evaluate_probs(global_test[1], predict(model, global_test[0])))
            m = evaluate_probs(d["test"][1], predict(model, d["test"][0]))
            if verbose:
                print(f"[{bank}] Round {r} Local PR-AUC: {m['pr_auc']:.4f}")
            res["per_client_history"].setdefault(f"round_{r}", {})[bank] = m
    for r in sorted(per_bank_global):
        for k in per_bank_global[r][0]:
            res["history_centralized"].setdefault(k, []).append(
                (r, float(np.nanmean([m[k] for m in per_bank_global[r]]))))
    return res
