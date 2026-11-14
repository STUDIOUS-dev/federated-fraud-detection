import streamlit as st

FLOW_DOT = r"""
digraph G {
  rankdir=TB;
  node [shape=box, style="rounded,filled", fillcolor="#eef2f7", fontname="Helvetica"];
  A [label="Bank A\nlocal data"];
  B [label="Bank B\nlocal data"];
  C [label="Bank C\nlocal data"];
  D [label="Bank D\nlocal data"];
  MA [label="Local training"];
  MB [label="Local training"];
  MC [label="Local training"];
  MD [label="Local training"];
  FL [label="FL coordinator\n(aggregation)", fillcolor="#dbe7f5"];
  G [label="Global model"];
  E [label="Evaluation & SHAP"];
  A -> MA; B -> MB; C -> MC; D -> MD;
  MA -> FL [label="model updates", fontsize=10];
  MB -> FL; MC -> FL; MD -> FL;
  FL -> G -> E;
}
"""


def render(res_svc, mod_svc, exp_svc):
    st.markdown("""
    ### Objective
    Simulate heterogeneous banks collaboratively training a fraud detector without sharing raw transaction data.

    ### Key Facts
    - **4 Simulated Bank Clients** with unique transaction distributions
    - **PyTorch Tabular MLP** for deep feature extraction
    - **Flower** as the Federated Learning framework
    - **FL Algorithms**: FedAvg, FedProx, FedMedian, Trimmed Mean, Krum
    - **SHAP** for transparent model attribution

    ### Architecture Flow
    """)
    st.graphviz_chart(FLOW_DOT)
    st.info(
        "Raw transaction data stays local throughout the simulation. Only model parameters (weights) "
        "are exchanged during federated training. No formal privacy guarantees (e.g. Differential "
        "Privacy) or Secure Aggregation are applied in this demonstration."
    )
