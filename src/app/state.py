import streamlit as st

def init_app_state():
    """Initializes the minimal application session state."""
    defaults = {
        "selected_experiment": "exp10_defense",
        "selected_transaction_idx": 0,
        "selected_view": "Home",
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def get_state(key: str, default=None):
    return st.session_state.get(key, default)

def set_state(key: str, value):
    st.session_state[key] = value
