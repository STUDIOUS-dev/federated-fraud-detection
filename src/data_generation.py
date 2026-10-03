from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import BANK_PROFILES, Config
def generate_bank_transactions(bank_id: str, cfg: Config) -> pd.DataFrame:
    if cfg.heterogeneity == "iid":
        profile = {"name": "Standardized", "amount_scale": 1.0, "tx_rate": 1.0, "fraud_rate": 0.010}
    else:
        profile = BANK_PROFILES[bank_id]
        
    rng = np.random.default_rng(cfg.seed + list(BANK_PROFILES).index(bank_id) * 1000)
    rows: list[dict] = []
    start = pd.Timestamp("2025-01-01", tz="UTC")
    
    n_customers = cfg.n_customers_per_bank
    if cfg.client_sizes is not None:
        idx = list(BANK_PROFILES.keys()).index(bank_id)
        n_customers = cfg.client_sizes[idx]

    for c in range(n_customers):
        customer_id = f"{bank_id}_cust_{c:05d}"
        # Stable customer spending baseline makes historical deviation features meaningful.
        customer_mean = float(rng.lognormal(mean=np.log(120 * profile["amount_scale"]), sigma=0.55))
        n_tx = max(8, int(rng.poisson(cfg.transactions_per_customer * profile["tx_rate"])))
        times = np.sort(rng.integers(0, 60 * 60 * 24 * 60, size=n_tx))
        merchant_ids = rng.integers(0, 3000, size=n_tx)

        for i, minutes in enumerate(times):
            amount = float(max(1.0, rng.lognormal(np.log(customer_mean), 0.60)))
            hour = int((minutes // 60) % 24)
            unusual_hour = hour in {1, 2, 3, 4}
            # Fraud probability is intentionally small and behavior-linked.
            score = -6.2
            score += 0.0035 * max(0.0, amount - customer_mean)
            score += 0.8 if unusual_hour else 0.0
            # Adjust overall fraud rate shift based on profile
            score += np.log(profile["fraud_rate"] / (1 - profile["fraud_rate"]) + 1e-9) + 4.6 # basic calibration
            
            fraud_prob = 1.0 / (1.0 + np.exp(-score))
            fraud = int(rng.random() < fraud_prob)
            if fraud and rng.random() < 0.45:
                amount *= rng.uniform(2.0, 5.0)

            rows.append({
                "bank_id": bank_id,
                "customer_id": customer_id,
                "terminal_id": f"term_{merchant_ids[i]:04d}",
                "timestamp": start + pd.Timedelta(minutes=int(minutes)),
                "amount": amount,
                "is_fraud": fraud,
            })

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def generate_federation(cfg: Config) -> dict[str, pd.DataFrame]:
    return {bank_id: generate_bank_transactions(bank_id, cfg) for bank_id in BANK_PROFILES}
