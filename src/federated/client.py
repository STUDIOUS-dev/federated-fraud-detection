import numpy as np
import flwr as fl
from src.config import BANK_PROFILES
from src.federated.local_training import train_local
from src.evaluation import evaluate_probs
from src.model import FraudMLP, numpy_to_state, state_to_numpy, predict


def _client_index(cid: str) -> int:
    banks = list(BANK_PROFILES)
    return banks.index(cid) if cid in banks else 0


class FraudClient(fl.client.NumPyClient):
    def __init__(self, cid: str, train_X, train_y, test_X, test_y, cfg, n_features):
        self.cid = cid
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.cfg = cfg
        self.n_features = n_features
        self.model = FraudMLP(n_features, cfg.hidden_dim, cfg.dropout)

    def get_parameters(self, config):
        return state_to_numpy(self.model)

    def set_parameters(self, parameters):
        numpy_to_state(self.model, parameters)

    def _train_seed(self, config) -> int:
        # Deterministic per (run seed, round, client) so Ray scheduling cannot change results.
        rnd = int(config.get("server_round", 0))
        return int(self.cfg.seed) * 100003 + rnd * 101 + _client_index(self.cid)

    def _train(self, parameters, config, labels=None):
        return train_local(
            initial_state=parameters,
            X=self.train_X,
            y=self.train_y if labels is None else labels,
            n_features=self.n_features,
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
            lr=self.cfg.lr,
            local_epochs=self.cfg.local_epochs,
            batch_size=self.cfg.batch_size,
            prox_mu=float(config.get("prox_mu", 0.0)),
            seed=self._train_seed(config),
        )

    def fit(self, parameters, config):
        new_parameters, num_examples = self._train(parameters, config)
        return new_parameters, num_examples, {"cid": self.cid}

    def evaluate(self, parameters, config):
        if not self.cfg.per_client_eval:
            return 0.0, len(self.test_y), {}

        self.set_parameters(parameters)
        metrics = evaluate_probs(self.test_y, predict(self.model, self.test_X))
        # Embed CID in metrics so server can map it
        metrics["cid"] = self.cid

        # Flower expects (loss, num_examples, metrics)
        return 0.0, len(self.test_y), metrics


class MaliciousFraudClient(FraudClient):
    """Compromised bank. Behaviour is selected by ``cfg.attack``:

    * ``scale``     - sends its trained weights multiplied by -10 (original Exp 9 attack).
    * ``signflip``  - sends global + (-5) * (local - global): an amplified, reversed update.
    * ``labelflip`` - trains honestly but on inverted fraud labels (data poisoning).
    """

    SCALE = -10.0
    FLIP_GAIN = -5.0

    def fit(self, parameters, config):
        attack = getattr(self.cfg, "attack", "scale")
        if attack == "labelflip":
            new_params, n = self._train(parameters, config, labels=1.0 - self.train_y)
            return new_params, n, {"cid": self.cid}

        new_params, n = self._train(parameters, config)
        if attack == "signflip":
            poisoned = [(g + self.FLIP_GAIN * (p - g)).astype(np.float32)
                        for g, p in zip(parameters, new_params)]
        elif attack == "scale":
            poisoned = [(p * self.SCALE).astype(np.float32) for p in new_params]
        else:
            raise ValueError(f"Unknown attack '{attack}'")
        return poisoned, n, {"cid": self.cid}


def get_numpy_client_fn(client_data: dict, cfg, n_features):
    """idx -> NumPyClient (honest or malicious). Shared by the Flower simulation and the in-process runner."""
    bank_ids = list(client_data.keys())
    malicious = cfg.malicious_set()

    def make(idx: int) -> FraudClient:
        bank_id = bank_ids[idx]
        data = client_data[bank_id]
        ClientClass = MaliciousFraudClient if idx in malicious else FraudClient
        return ClientClass(
            cid=bank_id,
            train_X=data["train"][0],
            train_y=data["train"][1],
            test_X=data["test"][0],
            test_y=data["test"][1],
            cfg=cfg,
            n_features=n_features,
        )
    return make


def get_client_fn(client_data: dict, cfg, n_features):
    make = get_numpy_client_fn(client_data, cfg, n_features)

    def client_fn(cid: str) -> fl.client.Client:
        # Flower passes the stringified partition index as cid
        return make(int(cid)).to_client()

    return client_fn
