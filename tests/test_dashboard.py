import pytest
from streamlit.testing.v1 import AppTest
import os
from pathlib import Path

def test_dashboard_smoke():
    # Ensure there's a summary.jsonl file so it doesn't just stop early
    assert os.path.exists("results/summary.jsonl"), "Summary file missing, cannot test dashboard properly"
    
    dashboard = Path(__file__).resolve().parents[1] / "src" / "dashboard.py"
    at = AppTest.from_file(str(dashboard))
    # Set a timeout and run the script
    at.run(timeout=60)
    
    # Assert no exceptions occurred during execution
    assert not at.exception, f"Dashboard crashed with exceptions: {at.exception}"
