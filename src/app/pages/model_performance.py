import streamlit as st
import pandas as pd
from src.app.labels import algorithm_label

def render(res_svc, mod_svc, exp_svc):
    st.header("Model Performance")
    
    summary = res_svc.load_summary()
    if not summary:
        st.warning("No summary data found.")
        return
        
    st.markdown("### Global Metrics Comparison")
    st.write("Comparing verified final results across all core experiments.")
    
    # Build comparison table
    records = []
    for r in summary:
        metrics = r.get("global_metrics", {})
        if metrics:
            records.append({
                "Experiment": r["exp_name"],
                "Aggregation": algorithm_label(r),
                "PR-AUC": metrics.get("pr_auc"),
                "F1": metrics.get("f1"),
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall")
            })
            
    df_global = pd.DataFrame(records)
    st.dataframe(df_global.round(4), use_container_width=True)
    
    if "PR-AUC" in df_global.columns:
        st.bar_chart(df_global.set_index("Experiment")["PR-AUC"])
        
    st.divider()
    
    st.markdown("### Per-Client Heterogeneity")
    st.write("Displays the performance on individual bank domains (where computed).")
    
    selected_exp = st.selectbox("Select Experiment for Client Breakdown", df_global["Experiment"])
    client_metrics = res_svc.get_client_metrics(selected_exp)
    
    if client_metrics:
        client_records = []
        for client, cm in client_metrics.items():
            client_records.append({**cm, "Bank": client})
            
        df_client = pd.DataFrame(client_records).set_index("Bank")
        st.dataframe(df_client.round(4), use_container_width=True)
        
        if "pr_auc" in df_client.columns:
            st.bar_chart(df_client["pr_auc"])
    else:
        st.info("No per-client metrics recorded for this experiment.")
