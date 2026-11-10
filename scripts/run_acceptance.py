import os
import sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

def test_pages():
    print("Initializing AppTest for dashboard.py...")
    at = AppTest.from_file("src/dashboard.py")
    at.run(timeout=30)
    assert not at.exception, f"App crashed on startup: {at.exception}"

    pages = [
        "Home",
        "Bank Simulation",
        "Federated Training",
        "Model Performance",
        "Robustness",
        "Transaction Analyzer",
        "Explainability",
        "Project Information"
    ]

    for page in pages:
        print(f"Navigating to {page}...")
        # In streamlit AppTest, we can set the sidebar radio value
        at.sidebar.radio[0].set_value(page).run(timeout=30)
        assert not at.exception, f"App crashed when navigating to {page}: {at.exception}"
        
        # Check specific constraints per page
        text_content = "".join([m.value for m in at.markdown])
        
        if page == "Bank Simulation":
            assert "BANK_A" in text_content or any("BANK_A" in getattr(c, 'value', '') for c in getattr(at, 'tabs', []))
            print("  - Bank metadata verified.")
            
        elif page == "Federated Training":
            # Just verify it loads and doesn't crash (AppTest passes if no exception)
            pass
            
        elif page == "Model Performance":
            assert "Global Metrics Comparison" in text_content or len(at.dataframe) > 0
            
        elif page == "Robustness":
            # Robustness page should display metrics for Clean FedAvg, Poisoned FedAvg, and Poisoned FedMedian
            all_text = " ".join([getattr(e, 'value', '') for e in at.markdown] + [str(getattr(e, 'label', '')) + " " + str(getattr(e, 'value', '')) for e in at.metric])
            from src.services.results import ResultsService
            svc = ResultsService()
            for exp in ("exp3_fedavg", "exp9_malicious", "exp10_defense"):
                val = svc.get_experiment_metrics(exp)["pr_auc"]
                assert f"{val:.4f}" in all_text, f"{exp} PR-AUC {val:.4f} missing from Robustness page"
            print("  - Robustness values verified against results/summary.jsonl.")

        elif page == "Project Information":
            all_text = " ".join([getattr(e, 'value', '') for e in at.markdown])
            assert "the problem" in all_text.lower() or "federated learning" in all_text.lower()
            assert "future work" in all_text.lower()
            print("  - Project Information content verified.")

def test_transaction_and_shap():
    print("Testing backend logic (Transaction Analyzer & SHAP)...")
    from src.services.model import ModelService
    from src.services.explainability import ExplainabilityService
    
    mod_svc = ModelService()
    mod_svc.load_model()
    
    assert mod_svc.model is not None, "Failed to load model"
    
    X = np.load("results/exp10_defense/test_sample_X.npy")
    y = np.load("results/exp10_defense/test_sample_y.npy")
    
    # Pick a transaction
    idx = 0
    raw_tx_df = mod_svc.get_test_transaction(idx)
    label = y[idx]
    
    # 1-4. Apply scaler and schema and check it matches test_sample_X
    X_tensor = mod_svc._preprocess(raw_tx_df)
    assert X_tensor.shape[1] == 14, f"Expected 14 features, got {X_tensor.shape[1]}"
    assert np.allclose(X_tensor.numpy(), X[idx].reshape(1, -1), atol=1e-5), "Preprocessing output does not match frozen test_sample_X.npy"
    print("  - Schema and scaler successfully applied, matches test_sample_X.npy.")
    
    # 5-8. Inference, probabilities, labels
    prob = mod_svc.predict_proba(raw_tx_df)[0]
    classification = mod_svc.predict(raw_tx_df)[0]
    assert 0 <= prob <= 1, f"Invalid probability: {prob}"
    print(f"  - Inference successful. Prob: {prob:.4f}, Classification: {classification}, Label: {label}")
    
    # SHAP
    exp_svc = ExplainabilityService(mod_svc)
    explanation = exp_svc.explain(raw_tx_df)
    
    assert explanation is not None
    assert explanation.get("base_value") is not None
    assert len(explanation["shap_values"]) == 14
    assert len(explanation["feature_names"]) == 14
    
    # Check that it generated a human readable summary
    summary = explanation["explanation"]
    assert len(summary) > 0
    print("  - SHAP feature attribution successful.")
    print("  - SHAP human explanation generated.")


def test_docs():
    print("Verifying documentation...")
    req_docs = [
        "README.md",
        "docs/FINAL_PRODUCT_GUIDE.md",
        "docs/DEMO_RUNBOOK.md",
        "docs/DEMO_CHECKLIST.md",
        "docs/FINAL_PRODUCT_PLAN.md",
        "PROJECT_PROGRESS.md"
    ]
    for d in req_docs:
        assert os.path.exists(d), f"Missing document: {d}"
    
    with open("docs/FINAL_PRODUCT_GUIDE.md", "r", encoding="utf-8") as f:
        content = f.read()
        assert "Streamlit" in content or "services" in content, "Guide doesn't describe the final UI architecture."
    print("  - Documentation exists and aligns with final product.")

if __name__ == "__main__":
    try:
        test_pages()
        test_transaction_and_shap()
        test_docs()
        print("\nALL ACCEPTANCE TESTS PASSED!")
    except Exception as e:
        print(f"\nACCEPTANCE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
