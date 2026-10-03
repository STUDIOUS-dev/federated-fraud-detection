# Final Product Guide

Welcome to the **Privacy-Aware & Explainable Federated Fraud Detection** dashboard.

This guide outlines how to navigate, interpret, and present the Streamlit application. The dashboard serves as a read-and-demonstrate interface over the pre-validated experimental results.

## 1. Installation & Launch
Ensure your virtual environment is active and dependencies are installed.
```bash
pip install -r requirements.txt
streamlit run src/dashboard.py
```
This will open the web interface in your default browser.

## 2. Page Navigation

### Home
Provides a high-level summary of the project's goal: collaboratively training a fraud detection model across four simulated banks without centralizing transaction data. The flowchart visually maps the Federated Learning process.

### Bank Simulation
Demonstrates data heterogeneity. Banks have drastically different profiles:
- **Bank A**: Retail-heavy, low fraud.
- **Bank C**: High-value, low transaction rate.
- **Bank D**: Regional small bank, high fraud rate.
These imbalances make standard training difficult, justifying the federated approach.

### Federated Training
Displays configuration metadata and progression of the global model across communication rounds. Select different algorithms (e.g., FedAvg vs FedProx) to observe stability across rounds.

### Model Performance
The core results table comparing Global PR-AUC, F1, Precision, and Recall across various scenarios (Class Imbalance, IID vs Non-IID, Client Dropout). A sub-section allows for inspecting specific metrics broken down by individual clients (Banks A, B, C, D).

### Robustness
A crucial demonstration of vulnerability and defense in FL:
- Shows the baseline performance of Clean FedAvg.
- Highlights catastrophic model collapse under data poisoning (`Poisoned FedAvg`).
- Demonstrates resilient recovery utilizing robust aggregation (`Poisoned FedMedian`).

### Transaction Analyzer
An interactive sandbox. 
1. Select a held-out test transaction via its ID.
2. View the exact input features that were presented to the model.
3. Observe the generated prediction and predicted probability alongside the true label.

### Explainability
Directly linked to the Transaction Analyzer. For the chosen transaction, this page unwraps the "black-box" model decision using SHAP (SHapley Additive exPlanations):
- Displays numerical SHAP attributions indicating which factors increased or decreased the fraud risk score.
- Translates the top numerical factors into a clean, human-readable semantic explanation.

### Project Information
Presents the underlying rationale, methodology, limitations, and future work for the project. Outlines the transition from centralized data silos to Federated Learning, its inherent security limitations (no Secure Aggregation), and planned expansions like Differential Privacy.

## 3. Data Flow & Integrity
The dashboard uses a dedicated internal Service Layer (`ResultsService`, `ModelService`, `ExplainabilityService`) to guarantee integrity:
- **No data regeneration for test sets**: It identically maps raw data to the specific `scaler.pkl` used in training, generating exact tensor matches with the frozen experiments.
- **Read-Only**: The UI does not overwrite or launch new long-running experiments by default, guaranteeing deterministic demonstrations.

## 4. Known Limitations
- The simulation runs sequentially on a single node rather than via distributed sockets.
- The UI relies on deterministic data cache warming upon the first load of the Transaction Analyzer.
- The model attributions strictly outline prediction correlations, not definitive causality.
