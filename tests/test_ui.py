import pytest
from streamlit.testing.v1 import AppTest

def test_app_initializes():
    """Smoke test to ensure the Streamlit dashboard initializes without crashing."""
    at = AppTest.from_file("src/dashboard.py")
    
    # Run the app
    at.run()
    
    assert not at.exception, f"App crashed with exceptions: {at.exception}"
    
    # Check that main title is rendered
    assert len(at.title) > 0
    assert "Privacy-Aware & Explainable Federated Fraud Detection" in at.title[0].value
