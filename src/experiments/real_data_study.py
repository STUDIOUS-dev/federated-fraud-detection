"""Separate study on REAL data (OpenML 'creditcard', id 1597). NOT part of the frozen synthetic results.

Setup: temporal split (first 80% rows train, last 20% test; row order = time order), train split
into 4 heterogeneous 'banks' (unequal sizes, skewed fraud share), then centralized / local / FedAvg /
FedMedian (+ FedAvg & FedMedian under one scaling attacker), mean +/- std over seeds.
Writes results/real_data/results.json and docs/REAL_DATA_RESULTS.md.

Run: python -m src.experiments.real_data_study [--seeds 3] [--rounds 10]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.evaluation import evaluate_probs
from src.federated.local_training import train_local, fedavg
from src.model import FraudMLP, numpy_to_state, state_to_numpy, predict

CSV = Path("data/real/creditcard.csv")
SIZES = [0.45, 0.30, 0.15, 0.10]
FRAUD_SKEW = [0.15, 0.20, 0.30, 0.35]  # share of fraud rows each bank gets (sums to 1.0)
HP = dict(hidden_dim=32, dropout=0.1, lr=1e-3, local_epochs=2, batch_size=256)
METRICS = ["pr_auc", "roc_auc", "recall_at_fpr1"]


def load():
    if not CSV.exists():
        from sklearn.datasets import fetch_openml
        CSV.parent.mkdir(parents=True, exist_ok=True)
        fetch_openml(data_id=1597, as_frame=True, parser="auto").frame.to_csv(CSV, index=False)
    df = pd.read_csv(CSV)
    return df.drop(columns=["Class"]).values.astype(np.float32), df["Class"].values.astype(np.float32)


def make_banks(Xtr, ytr, rng):
    """Random assignment with unequal sizes and an unequal fraud share per bank."""
    idx_f, idx_n = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
    rng.shuffle(idx_f); rng.shuffle(idx_n)
    cf = (np.cumsum(FRAUD_SKEW)[:-1] * len(idx_f)).astype(int)
    cn = (np.cumsum(SIZES)[:-1] * len(idx_n)).astype(int)
    parts = [np.concatenate([a, b]) for a, b in zip(np.split(idx_f, cf), np.split(idx_n, cn))]
    return [(Xtr[p], ytr[p]) for p in parts]


def aggregate(states, sizes, how):
    if how == "median":
        return [np.median(np.stack(l), axis=0).astype(np.float32) for l in zip(*states)]
    return fedavg(states, sizes)


def run_fed(banks, Xte, yte, n, seed, rounds, how, attacker=False):
    torch_seed(seed)
    g = state_to_numpy(FraudMLP(n, HP["hidden_dim"], HP["dropout"]))
    model = FraudMLP(n, HP["hidden_dim"], HP["dropout"])
    for r in range(rounds):
        states, sizes = [], []
        for i, (X, y) in enumerate(banks):
            s, k = train_local(g, X, y, n, seed=seed * 100003 + r * 101 + i, **HP)
            if attacker and i == 0:
                s = [g_ + 10.0 * (a - g_) for g_, a in zip(g, s)]  # scale x10 update
            states.append(s); sizes.append(k)
        g = aggregate(states, sizes, how)
    numpy_to_state(model, g)
    return evaluate_probs(yte, predict(model, Xte))


def torch_seed(s):
    import torch; torch.manual_seed(s)


def run_central(banks, Xte, yte, n, seed, rounds):
    X = np.vstack([b[0] for b in banks]); y = np.concatenate([b[1] for b in banks])
    torch_seed(seed)
    g = state_to_numpy(FraudMLP(n, HP["hidden_dim"], HP["dropout"]))
    for r in range(rounds):
        g, _ = train_local(g, X, y, n, seed=seed * 100003 + r * 101 + 50, **HP)
    m = FraudMLP(n, HP["hidden_dim"], HP["dropout"]); numpy_to_state(m, g)
    return evaluate_probs(yte, predict(m, Xte))


def run_local(banks, Xte, yte, n, seed, rounds):
    out = []
    for i, (X, y) in enumerate(banks):
        torch_seed(seed)
        g = state_to_numpy(FraudMLP(n, HP["hidden_dim"], HP["dropout"]))
        for r in range(rounds):
            g, _ = train_local(g, X, y, n, seed=seed * 100003 + r * 101 + i, **HP)
        m = FraudMLP(n, HP["hidden_dim"], HP["dropout"]); numpy_to_state(m, g)
        out.append(evaluate_probs(yte, predict(m, Xte)))
    return {k: float(np.nanmean([o[k] for o in out])) for k in out[0]}


def main(seeds=3, rounds=10):
    X, y = load()
    cut = int(len(X) * 0.8)
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr).astype(np.float32), sc.transform(Xte).astype(np.float32)
    print(f"train {len(Xtr)} ({int(ytr.sum())} fraud) | test {len(Xte)} ({int(yte.sum())} fraud)")
    scen = {
        "Centralized": lambda b, s: run_central(b, Xte, yte, X.shape[1], s, rounds),
        "Local (avg of banks)": lambda b, s: run_local(b, Xte, yte, X.shape[1], s, rounds),
        "FedAvg": lambda b, s: run_fed(b, Xte, yte, X.shape[1], s, rounds, "avg"),
        "FedMedian": lambda b, s: run_fed(b, Xte, yte, X.shape[1], s, rounds, "median"),
        "FedAvg + attacker": lambda b, s: run_fed(b, Xte, yte, X.shape[1], s, rounds, "avg", True),
        "FedMedian + attacker": lambda b, s: run_fed(b, Xte, yte, X.shape[1], s, rounds, "median", True),
    }
    runs = {k: [] for k in scen}
    for s in range(seeds):
        banks = make_banks(Xtr, ytr, np.random.default_rng(s))
        for name, fn in scen.items():
            m = fn(banks, s); runs[name].append(m)
            print(f"seed {s} {name:22s} PR-AUC {m['pr_auc']:.3f}", flush=True)
    summary = {n: {k: (float(np.nanmean([r[k] for r in rs])), float(np.nanstd([r[k] for r in rs])))
                   for k in METRICS} for n, rs in runs.items()}
    Path("results/real_data").mkdir(parents=True, exist_ok=True)
    json.dump({"seeds": seeds, "rounds": rounds, "test_frauds": int(yte.sum()), "summary": summary, "runs": runs},
              open("results/real_data/results.json", "w"), indent=2)
    lines = ["# Real-data study (OpenML creditcard)", "",
             "Separate from the frozen synthetic results. Temporal split by row order, 4 simulated banks "
             f"(unequal sizes, skewed fraud share), {seeds} seeds, {rounds} rounds. Test set has only "
             f"{int(yte.sum())} frauds, so differences are noisy. The banks are a simulated partition of one "
             "dataset, not real institutions.", "",
             "| Setup | PR-AUC | ROC-AUC | Recall@1%FPR |", "|---|---|---|---|"]
    for n, m in summary.items():
        lines.append(f"| {n} | " + " | ".join(f"{m[k][0]:.3f} ± {m[k][1]:.3f}" for k in METRICS) + " |")
    Path("docs/REAL_DATA_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=10)
    a = ap.parse_args(); main(a.seeds, a.rounds)
