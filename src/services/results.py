import json
from pathlib import Path
import math

class ResultsService:
    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self.summary_path = self.base_dir / "summary.jsonl"
        
    def load_summary(self) -> list[dict]:
        """Loads and parses the summary.jsonl file."""
        if not self.summary_path.exists():
            raise FileNotFoundError(f"Summary file not found at {self.summary_path}")
            
        records = []
        with open(self.summary_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError:
                        continue # Safely skip malformed lines
                        
        # Filter duplicates keeping the latest
        unique_records = {}
        for r in records:
            if "exp_name" in r:
                unique_records[r["exp_name"]] = r
                
        return list(unique_records.values())

    def get_experiment(self, exp_name: str) -> dict:
        """Gets a specific experiment's summary record."""
        summary = self.load_summary()
        for record in summary:
            if record.get("exp_name") == exp_name:
                return record
        raise ValueError(f"Experiment '{exp_name}' not found in summary.")

    def get_experiment_metrics(self, exp_name: str) -> dict:
        """Gets global metrics for a specific experiment, handling missing/invalid values."""
        record = self.get_experiment(exp_name)
        metrics = record.get("global_metrics", {})
        
        # Clean NaNs and handle missing
        cleaned = {}
        for k, v in metrics.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cleaned[k] = None
            else:
                cleaned[k] = v
        return cleaned

    def get_client_metrics(self, exp_name: str) -> dict:
        """Gets per-client metrics for an experiment."""
        record = self.get_experiment(exp_name)
        per_client = record.get("per_client_metrics", {})
        
        # Clean NaNs
        cleaned = {}
        for client, metrics in per_client.items():
            cleaned_metrics = {}
            for k, v in metrics.items():
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    cleaned_metrics[k] = None
                else:
                    cleaned_metrics[k] = v
            cleaned[client] = cleaned_metrics
            
        return cleaned

    def get_round_history(self, exp_name: str) -> dict:
        """Loads detailed round history from results.json."""
        exp_dir = self.base_dir / exp_name
        results_file = exp_dir / "results.json"
        
        if not results_file.exists():
            raise FileNotFoundError(f"Detailed results not found for {exp_name}")
            
        with open(results_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"Malformed results.json in {exp_name}")
                
        return {
            "history_centralized": data.get("history_centralized", {}),
            "per_client_history": data.get("per_client_history", {}),
            "participating_clients": data.get("participating_clients", {})
        }
