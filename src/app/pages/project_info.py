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
    - **FedMedian**: A robust aggregation method that takes the coordinate-wise median of updates, which mitigated the simple poisoning attacks tested here.
    - **Trimmed Mean (FedTrimmedAvg)**: Averages updates after discarding the most extreme values in each coordinate.
    - **Krum**: Selects the single client update closest to its neighbours, discarding outliers.

    ### Limitations
    - **Simulation Only**: The main study uses synthetic data with a weak fraud signal (about 53 frauds per test pool, so per-bank results are very noisy). Banks are simulated on one machine; no real network cost is measured.
    - **No Formal Privacy Guarantees**: While data remains local, techniques like Differential Privacy (DP) and Secure Aggregation (SecAgg) are not implemented. Gradient inversion attacks could theoretically leak information.
    - **Attack Coverage**: Three simple, non-adaptive poisoning attacks (scale, sign-flip, label-flip) with at most 2 of 4 malicious clients. Adaptive or colluding attackers were not evaluated.
    - **Study Scope**: A 38-scenario x 5-seed synthetic study, plus a separate real-data study on the public OpenML creditcard dataset. Its "banks" are a random, skewed partition of one dataset, not real institutions, and its test set has only 75 frauds, so differences between healthy methods are within noise.
    - **Mild Heterogeneity**: Null results for IID-ness and FedProx may not carry over to real data.
    - **Not a production system.**

    ### Future Work
    - Implement **Differential Privacy** to bound information leakage from weight updates.
    - Implement **Secure Aggregation** using cryptographic masking.
    - Scale to more clients and evaluate actual communication/latency costs.
    - Test against stronger, adaptive and colluding Byzantine attacks.
    """)
