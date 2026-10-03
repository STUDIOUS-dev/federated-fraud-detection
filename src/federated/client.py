import flwr as fl
from src.federated.local_training import train_local
from src.evaluation import evaluate_probs
from src.model import FraudMLP, numpy_to_state, state_to_numpy, predict

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

    def fit(self, parameters, config):
        # config can pass dynamic mu, etc.
        mu = float(config.get("prox_mu", 0.0))
        
        new_parameters, num_examples = train_local(
            initial_state=parameters,
            X=self.train_X,
            y=self.train_y,
            n_features=self.n_features,
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
            lr=self.cfg.lr,
            local_epochs=self.cfg.local_epochs,
            batch_size=self.cfg.batch_size,
            prox_mu=mu
        )
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
    def fit(self, parameters, config):
        new_params, num_examples, metrics = super().fit(parameters, config)
        # Poisoning strategy: Scale weights significantly to destroy global model
        poisoned = [p * -10.0 for p in new_params]
        return poisoned, num_examples, metrics

def get_client_fn(client_data: dict, cfg, n_features):
    # Create mapping from integer index to bank_id since Flower passes stringified ints as cid
    bank_ids = list(client_data.keys())
    
    def client_fn(cid: str) -> fl.client.Client:
        idx = int(cid)
        bank_id = bank_ids[idx]
        data = client_data[bank_id]
        
        is_malicious = (idx == cfg.malicious_client_index)
        ClientClass = MaliciousFraudClient if is_malicious else FraudClient
        
        return ClientClass(
            cid=bank_id,
            train_X=data["train"][0],
            train_y=data["train"][1],
            test_X=data["test"][0],
            test_y=data["test"][1],
            cfg=cfg,
            n_features=n_features
        ).to_client()
        
    return client_fn
