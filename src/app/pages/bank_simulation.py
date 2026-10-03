import streamlit as st
import pandas as pd
from src.config import BANK_PROFILES

def render(res_svc, mod_svc, exp_svc):
    st.header("Bank Simulation Heterogeneity")
    st.write("This simulation involves four highly heterogeneous financial institutions.")
    
    cols = st.columns(4)
    for i, (b_id, profile) in enumerate(BANK_PROFILES.items()):
        with cols[i]:
            st.markdown(f"### {b_id.upper()}")
            st.markdown(f"**Type:** {profile['name']}")
            st.metric("Amount Scale", f"{profile['amount_scale']}x")
            st.metric("Tx Rate", f"{profile['tx_rate']}x")
            st.metric("Fraud Rate", f"{profile['fraud_rate']:.1%}")
            
    st.divider()
    
    # Use the test set from ModelService to show empirical data
    if mod_svc.model is not None:
        try:
            df = mod_svc.get_test_transactions()
            st.subheader("Empirical Test Data Distributions")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Transaction Volume per Bank**")
                vol_counts = df['bank_id'].value_counts().reset_index()
                st.bar_chart(vol_counts, x="bank_id", y="count", use_container_width=True)
                
            with col2:
                st.write("**Fraud Cases per Bank**")
                fraud_counts = df[df['is_fraud'] == 1]['bank_id'].value_counts().reset_index()
                st.bar_chart(fraud_counts, x="bank_id", y="count", use_container_width=True)
                
            st.write("**Transaction Amount Distribution**")
            # Sample to avoid heavy rendering
            scatter_df = df.sample(min(2000, len(df)))
            st.scatter_chart(scatter_df, x="bank_id", y="amount", color="is_fraud")
                
        except Exception as e:
            st.warning(f"Could not load empirical data: {e}")
