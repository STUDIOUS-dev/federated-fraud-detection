# Demo Runbook: Federated Fraud Detection

## Before the demo
### Environment Setup
1. Activate the Python virtual environment:
   ```bash
   # (assuming standard venv)
   .\venv\Scripts\activate
   ```
2. Verify all dependencies are installed from `requirements.txt`.
3. Check that the final metrics exist in `results/summary.jsonl`.
4. Ensure the pre-trained FedMedian model and test sets exist in `results/exp10_defense/`:
   - `model.pt`
   - `scaler.pkl`
   - `test_sample_X.npy`
   - `test_sample_y.npy`

### Commands
Start the Streamlit dashboard:
```bash
streamlit run src/dashboard.py
```

### Expected Startup Behavior
- The Streamlit app should open automatically in your browser at `http://localhost:8501`.
- The dashboard should have 5 tabs: Overview, Client/Heterogeneity, FL Training, Robustness, Explainability.
- If it fails, rely on the backup screenshots (if generated previously).

---

## Demo Flow (5–8 minutes)

### 1. Project Overview & Centralized Baseline (1 min)
- **Action**: Open the **Overview** tab.
- **Show**: The latest experiment metrics and the table of all experiments.
- **Talking Point**: "Welcome to our Federated Fraud Detection project. We simulated four heterogeneous banks (A, B, C, D). Initially, if we pooled all their data into a Centralized model, we achieved a PR-AUC of **0.2508**. Our goal is to see if we can match this using Federated Learning without sharing raw data."

### 2. Federated Learning Workflow & Clean FedAvg (1 min)
- **Action**: Go to the **FL Training** tab. Select `exp3_fedavg` from the dropdown.
- **Show**: The Global PR-AUC over Rounds chart.
- **Talking Point**: "Here is our standard Federated Averaging (FedAvg) training over 5 rounds. We reach a global PR-AUC of **0.2439**, which is highly comparable to the centralized baseline. The banks collaborate effectively while maintaining data privacy."

### 3. Client Heterogeneity (1 min)
- **Action**: Go to the **Client/Heterogeneity** tab.
- **Show**: The bar chart comparing PR-AUC across Bank A, B, C, and D for different experiments.
- **Talking Point**: "However, not all banks are equal. We injected realistic data heterogeneity (class imbalance, size variations). For instance, Bank C has much better local performance due to larger data size, while others struggle locally but benefit greatly from the global model."

### 4. Poisoned FedAvg vs FedMedian Mitigation (2 mins)
- **Action**: Go to the **Robustness** tab.
- **Show**: The comparison chart between `exp3_fedavg`, `exp9_malicious`, and `exp10_defense`.
- **Talking Point**: "What happens if a bank is compromised? We introduced a malicious client that sends inverted model weights (Poisoned FedAvg). The global PR-AUC collapses from 0.2439 to **0.0036**. 
To mitigate this, we deployed **FedMedian**, a robust aggregation strategy. As you can see, FedMedian successfully ignores the poisoned updates and recovers the performance back to **0.2435**."

### 5. Explainability with SHAP (1.5 mins)
- **Action**: Go to the **Explainability (SHAP)** tab. Move the transaction index slider.
- **Show**: A transaction being predicted, along with the "Human-Readable Explanation" text area.
- **Talking Point**: "Finally, in the financial sector, black-box models aren't enough. We integrated SHAP to explain individual transaction predictions from our final FedMedian model. For this transaction, the model predicted it as [Fraud/Legitimate], and our human-readable explanation highlights exactly which features (like recent transaction amount or location) drove this decision."

### 6. Conclusion (0.5 mins)
- **Talking Point**: "In conclusion, we successfully demonstrated that Federated Learning can match centralized performance for fraud detection, while FedMedian ensures robustness against malicious attacks, and SHAP provides the necessary regulatory transparency."

---

## Key Numerical Results (Verified)
- **Centralized baseline PR-AUC (exp1)**: 0.2508
- **Clean FedAvg PR-AUC (exp3)**: 0.2439
- **Poisoned FedAvg PR-AUC (exp9)**: 0.0036
- **Poisoned FedMedian PR-AUC (exp10)**: 0.2435

---

## Backup Plan
If the live environment or Streamlit dashboard fails during the demo:
1. **Fallback 1 (Results)**: Open `scratch/generate_final_report.py` and run it via `python scratch/generate_final_report.py` in the terminal to display the markdown table of results directly in the console.
2. **Fallback 2 (Explainability)**: Run a manual python script (or reference `src/explainability.py`) to generate a print-out of the SHAP explanation to show the audience.
3. **Fallback 3 (Visuals)**: Have pre-captured screenshots of the 5 dashboard tabs saved in a `docs/assets/` folder to present as a slideshow.
