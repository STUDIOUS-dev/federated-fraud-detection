from dataclasses import dataclass
import yaml
from pathlib import Path

BANK_PROFILES = {
    "bank_a": {"name": "Retail-heavy", "amount_scale": 1.0, "tx_rate": 1.0, "fraud_rate": 0.006},
    "bank_b": {"name": "E-commerce-heavy", "amount_scale": 1.15, "tx_rate": 1.25, "fraud_rate": 0.010},
    "bank_c": {"name": "High-value", "amount_scale": 2.0, "tx_rate": 0.65, "fraud_rate": 0.004},
    "bank_d": {"name": "Regional-small", "amount_scale": 0.8, "tx_rate": 0.50, "fraud_rate": 0.015},
}

@dataclass
class Config:
    seed: int
    n_customers_per_bank: int
    transactions_per_customer: int
    test_fraction: float
    heterogeneity: str
    
    hidden_dim: int
    dropout: float
    lr: float
    
    local_epochs: int
    batch_size: int
    
    clients: int
    rounds: int
    aggregation: str
    fedprox_mu: float
    fraction_fit: float
    
    global_eval: bool
    per_client_eval: bool
    
    # Advanced FL properties
    client_sizes: list[int] | None = None
    dropout_rate: float = 0.0
    malicious_client_index: int = -1
    # Phase C additions (all optional / backwards compatible)
    malicious_clients: list[int] | None = None   # overrides malicious_client_index when set
    attack: str = "scale"                        # scale | signflip | labelflip
    trim_beta: float = 0.25                      # FedTrimmedAvg trim fraction per side

    def malicious_set(self) -> set[int]:
        if self.malicious_clients:
            return set(self.malicious_clients)
        return {self.malicious_client_index} if self.malicious_client_index >= 0 else set()

def load_config(config_path: str = "configs/default.yaml") -> Config:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
        
    return Config(
        seed=raw["data"]["seed"],
        n_customers_per_bank=raw["data"]["n_customers_per_bank"],
        transactions_per_customer=raw["data"]["transactions_per_customer"],
        test_fraction=raw["data"]["test_fraction"],
        heterogeneity=raw["data"].get("heterogeneity", "default"),
        
        hidden_dim=raw["model"]["hidden_dim"],
        dropout=raw["model"]["dropout"],
        lr=raw["model"]["lr"],
        
        local_epochs=raw["training"]["local_epochs"],
        batch_size=raw["training"]["batch_size"],
        
        clients=raw["federated"]["clients"],
        rounds=raw["federated"]["rounds"],
        aggregation=raw["federated"]["aggregation"],
        fedprox_mu=raw["federated"]["fedprox_mu"],
        fraction_fit=raw["federated"]["fraction_fit"],
        
        client_sizes=raw["data"].get("client_sizes", None),
        dropout_rate=raw["federated"].get("dropout_rate", 0.0),
        malicious_client_index=raw["federated"].get("malicious_client_index", -1),
        malicious_clients=raw["federated"].get("malicious_clients", None),
        attack=raw["federated"].get("attack", "scale"),
        trim_beta=raw["federated"].get("trim_beta", 0.25),
        
        global_eval=raw["evaluation"]["global_eval"],
        per_client_eval=raw["evaluation"]["per_client_eval"],
    )
