import streamlit as st

def render(res_svc, mod_svc, exp_svc):
    st.header("Project Information")
    
    st.markdown("""
    ### The Problem
    Detecting financial fraud requires large, diverse datasets. However, banks cannot freely share 
    raw customer transaction data due to strict privacy regulations, competitive secrecy, and security risks. 
    This creates isolated "data silos," limiting the performance of any individual bank's fraud detection system.

    ### Why Federated Learning (FL)?
    FL allows multiple institutions to collaboratively train a shared global model without ever pooling their raw data centrally.
    Only model parameters (weights/gradients) are exchanged. This preserves data locality while benefiting from the collective intelligence of all participants.

    ### Aggregation Strategies Used
    - **FedAvg**: The standard algorithm. It computes a sample-size weighted average of local models.
    - **FedProx**: Adds a proximal term to local training, preventing local models from drifting too far from the global model, which is helpful under high heterogeneity.
    - **FedMedian**: A robust aggregation method that takes the coordinate-wise median of updates, heavily mitigating the impact of poisoned updates from malicious clients.

    ### Limitations
    - **Simulation Only**: The current implementation runs on synthetic/controlled data and simulates network clients locally.
    - **No Formal Privacy Guarantees**: While data remains local, techniques like Differential Privacy (DP) and Secure Aggregation (SecAgg) are not implemented. Gradient inversion attacks could theoretically leak information.
    - **Attack Coverage**: The robustness evaluation is limited to a single specific data-poisoning scenario.

    ### Future Work
    - Implement **Differential Privacy** to bound information leakage from weight updates.
    - Implement **Secure Aggregation** using cryptographic masking.
    - Scale to more clients and evaluate actual communication/latency costs.
    - Test against stronger, coordinated Byzantine attacks.
    """)
