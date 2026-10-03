import streamlit as st

def render(res_svc, mod_svc, exp_svc):
    st.markdown("""
    ### Objective
    Simulate heterogeneous banks collaboratively training a fraud detector without sharing raw transaction data.
    
    ### Key Facts
    - **4 Simulated Bank Clients** with unique transaction distributions
    - **PyTorch Tabular MLP** for deep feature extraction
    - **Flower** as the Federated Learning framework
    - **FL Algorithms**: FedAvg, FedProx, FedMedian
    - **SHAP** for transparent model attribution
    
    ### Architecture Flow
    ```mermaid
    graph TD
        subgraph Bank Clients
        A[Bank A Local Data]
        B[Bank B Local Data]
        C[Bank C Local Data]
        D[Bank D Local Data]
        end
        
        subgraph Local Training
        A --> MA[Train Local Model]
        B --> MB[Train Local Model]
        C --> MC[Train Local Model]
        D --> MD[Train Local Model]
        end
        
        MA -->|Model Updates| FL[FL Coordinator]
        MB -->|Model Updates| FL
        MC -->|Model Updates| FL
        MD -->|Model Updates| FL
        
        FL --> G[Global Model]
        G --> E[Evaluation & SHAP]
    ```
    
    > [!NOTE]
    > Raw transaction data strictly remains local throughout the simulation. Only model parameters (weights) are exchanged during the federated training process. Formal privacy guarantees (e.g. Differential Privacy) are not currently applied in this demonstration.
    """)
