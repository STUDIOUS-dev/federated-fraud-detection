import streamlit as st
import pandas as pd
import ast

def render(res_svc, mod_svc, exp_svc):
    st.header("Federated Training")
    
    summary = res_svc.load_summary()
    exp_names = [r["exp_name"] for r in summary if "fed" in r.get("aggregation", "").lower() or r.get("aggregation") == "FedAvg" or "exp" in r.get("exp_name")]
    
    if not exp_names:
        st.warning("No federated experiments found in the results.")
        return
        
    selected_exp = st.selectbox("Select Experiment to View History", exp_names)
    exp = res_svc.get_experiment(selected_exp)
    
    st.markdown("### Training Configuration")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algorithm", exp.get("aggregation", "N/A"))
    c2.metric("Rounds", exp.get("rounds", "N/A"))
    c3.metric("Clients Participating", exp.get("clients", "N/A"))
    c4.metric("Local Epochs", exp.get("local_epochs", "N/A"))
    
    try:
        history = res_svc.get_round_history(selected_exp)
        
        pr_auc_history = history.get("history_centralized", {}).get("pr_auc", [])
        if pr_auc_history:
            df_hist = pd.DataFrame(pr_auc_history, columns=["Round", "PR-AUC"]).set_index("Round")
            st.markdown("### Global PR-AUC over Rounds")
            st.line_chart(df_hist)
        else:
            st.info("No round-by-round global PR-AUC history found for this experiment.")
            
        participants = history.get("participating_clients", {})
        if participants:
            st.markdown("### Client Participation")
            st.json(participants)
            
    except Exception as e:
        st.error(f"Could not load detailed history: {e}")
