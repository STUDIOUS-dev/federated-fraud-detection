from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

def test_app_initializes():
    """Smoke test to ensure the Streamlit dashboard initializes without crashing."""
    dashboard = Path(__file__).resolve().parents[1] / "src" / "dashboard.py"
    at = AppTest.from_file(str(dashboard))

    # Run the app (generous timeout: cold start imports torch/shap/pandas and
    # can take well over the 3 s default, and >60 s on a slow disk)
    at.run(timeout=180)
    
    assert not at.exception, f"App crashed with exceptions: {at.exception}"
    
    # Check that main title is rendered
    assert len(at.title) > 0
    assert "Privacy-Aware & Explainable Federated Fraud Detection" in at.title[0].value
