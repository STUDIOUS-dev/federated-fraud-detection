import sys
from pathlib import Path

# Make `src.*` importable when launched as `streamlit run src/dashboard.py`
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
from src.app.state import init_app_state, get_state, set_state
from src.services.results import ResultsService
from src.services.model import ModelService
from src.services.explainability import ExplainabilityService

# Set page config
st.set_page_config(
    page_title="Federated Fraud Detection", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Initialize application state
init_app_state()

@st.cache_resource
def get_results_service():
    return ResultsService()

@st.cache_resource
def get_model_service():
    svc = ModelService()
    try:
        svc.load_model()
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
    return svc

@st.cache_resource
def get_explainability_service(_model_service):
    if _model_service.model is None:
        return None
    return ExplainabilityService(_model_service)

# Load services
res_svc = get_results_service()
mod_svc = get_model_service()
exp_svc = get_explainability_service(mod_svc)

# Import pages
from src.app.pages import (
    home,
    bank_simulation,
    federated_training,
    model_performance,
    robustness,
    transaction_analyzer,
    live_stream,
    explainability,
    project_info
)

PAGES = {
    "Home": home.render,
    "Bank Simulation": bank_simulation.render,
    "Federated Training": federated_training.render,
    "Model Performance": model_performance.render,
    "Robustness": robustness.render,
    "Transaction Analyzer": transaction_analyzer.render,
    "Live Stream": live_stream.render,
    "Explainability": explainability.render,
    "Project Information": project_info.render,
}

# Sidebar navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
set_state("selected_view", selection)

# Render selected page
st.title("Privacy-Aware & Explainable Federated Fraud Detection")
st.markdown("##### *Collaborative Fraud Detection Across Heterogeneous Financial Institutions*")
st.divider()

page_function = PAGES[selection]
page_function(res_svc, mod_svc, exp_svc)
