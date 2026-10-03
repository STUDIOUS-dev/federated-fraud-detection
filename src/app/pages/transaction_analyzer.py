import streamlit as st
import pandas as pd
from src.app.state import get_state, set_state

def render(res_svc, mod_svc, exp_svc):
    st.header("Transaction Analyzer")
    
    if mod_svc.model is None:
        st.error("Model is not loaded. Cannot analyze transactions.")
        return
        
    try:
        df = mod_svc.get_test_transactions()
        total_tx = len(df)
        
        st.write(f"Loaded **{total_tx}** frozen held-out test transactions for inference.")
        
        current_idx = get_state("selected_transaction_idx", 0)
        
        selected_idx = st.number_input(
            "Select Transaction Index", 
            min_value=0, 
            max_value=total_tx - 1, 
            value=current_idx,
            step=1
        )
        
        # Save state so Explainability page can use the exact same transaction
        set_state("selected_transaction_idx", selected_idx)
        
        tx_df = mod_svc.get_test_transaction(selected_idx)
        
        st.markdown(f"### Transaction Details (ID: {selected_idx})")
        
        # Split into columns for neat presentation
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Raw Input Features**")
            st.dataframe(tx_df.drop(columns=["is_fraud"], errors="ignore").T, use_container_width=True)
            
        with col2:
            st.write("**Prediction**")
            
            prob = mod_svc.predict_proba(tx_df)[0]
            pred = mod_svc.predict(tx_df)[0]
            actual = tx_df["is_fraud"].iloc[0]
            
            st.metric("Model Probability", f"{prob:.2%}")
            
            st.markdown(f"**Predicted Class:** `{'FRAUD' if pred == 1 else 'LEGITIMATE'}`")
            st.markdown(f"**Actual Label:** `{'FRAUD' if actual == 1 else 'LEGITIMATE'}`")
            
            if pred == actual:
                st.success("✅ Correct Prediction")
            else:
                st.error("❌ Incorrect Prediction")
                
            st.info("Switch to the **Explainability** page to understand why the model made this prediction.")
            
    except Exception as e:
        st.error(f"Error loading transaction data: {e}")
