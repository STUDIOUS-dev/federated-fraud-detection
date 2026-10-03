import streamlit as st
import pandas as pd
from src.app.state import get_state

def render(res_svc, mod_svc, exp_svc):
    st.header("Explainability (SHAP)")
    
    if exp_svc is None:
        st.error("Explainability Service is not initialized. Check model loading.")
        return
        
    try:
        selected_idx = get_state("selected_transaction_idx", 0)
        st.write(f"Explaining Transaction ID: **{selected_idx}**")
        
        tx_df = mod_svc.get_test_transaction(selected_idx)
        
        with st.spinner("Generating explanation..."):
            explanation = exp_svc.explain(tx_df)
            
        st.markdown("### Prediction Summary")
        prob = explanation["model_probability"]
        st.metric("Fraud Probability", f"{prob:.2%}")
        st.markdown(f"**Prediction:** `{'FRAUD' if prob >= 0.5 else 'LEGITIMATE'}`")
        
        st.divider()
        
        st.markdown("### Human Explanation")
        st.info(explanation["explanation"])
        
        st.markdown("---")
        
        st.markdown("### SHAP Attribution Details")
        st.write("> **Note:** SHAP provides model-attribution explanations. It does not establish causal relationships.")
        
        # Construct DataFrame of SHAP values
        shap_df = pd.DataFrame({
            "Feature": explanation["feature_names"],
            "SHAP Value": explanation["shap_values"]
        })
        
        shap_df["Impact"] = shap_df["SHAP Value"].apply(lambda x: "Increased Fraud Risk" if x > 0 else "Decreased Fraud Risk")
        shap_df["Absolute Magnitude"] = shap_df["SHAP Value"].abs()
        
        # Sort by impact
        shap_df = shap_df.sort_values(by="Absolute Magnitude", ascending=False).drop(columns=["Absolute Magnitude"])
        
        st.dataframe(shap_df.style.map(
            lambda val: 'color: red' if val == "Increased Fraud Risk" else 'color: green', 
            subset=['Impact']
        ), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating explanation: {e}")
