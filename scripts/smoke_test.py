from src.services.results import ResultsService
from src.services.model import ModelService
from src.services.explainability import ExplainabilityService

def run_smoke_test():
    print("--- Running Smoke Test ---")
    
    # 1. Results Service
    print("1. Testing Results Service...")
    res_svc = ResultsService()
    summary = res_svc.load_summary()
    print(f"Loaded {len(summary)} experiments from summary.jsonl")
    
    exp10 = res_svc.get_experiment("exp10_defense")
    print(f"exp10_defense found. Config: {exp10['aggregation']}")
    
    metrics = res_svc.get_experiment_metrics("exp10_defense")
    print(f"Global PR-AUC: {metrics.get('pr_auc')}")
    
    history = res_svc.get_round_history("exp10_defense")
    print(f"History rounds recorded: {len(history['history_centralized'].get('pr_auc', []))}")
    
    # 2. Model Service
    print("\n2. Testing Model Service...")
    mod_svc = ModelService(exp_name="exp10_defense", base_dir="results")
    mod_svc.load_model()
    print("Model and Scaler loaded.")
    
    metadata = mod_svc.get_model_metadata()
    print(f"Model expects {metadata['n_features']} features.")
    
    df = mod_svc.get_test_transactions()
    print(f"Loaded test transactions. Shape: {df.shape}")
    
    tx = mod_svc.get_test_transaction(42)
    print("Selected transaction index 42.")
    
    prob = mod_svc.predict_proba(tx)
    print(f"Prediction Probability: {prob[0]}")
    
    # 3. Explainability Service
    print("\n3. Testing Explainability Service...")
    expl_svc = ExplainabilityService(mod_svc)
    explanation = expl_svc.explain(tx)
    print("Explanation successful.")
    print(f"Model Probability: {explanation['model_probability']}")
    print(f"Explanation Preview: {explanation['explanation'][:100]}...")
    
    print("\nSmoke test passed.")

if __name__ == "__main__":
    run_smoke_test()
