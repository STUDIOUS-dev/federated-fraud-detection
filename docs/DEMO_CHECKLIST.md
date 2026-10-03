# Demo Validation Checklist

## Environment Check
- [ ] Python virtual environment is active.
- [ ] Required dependencies from `requirements.txt` are installed.
- [ ] Project structure is intact.

## Test Suite
- [ ] Run `python -m pytest -v`
- [ ] All tests pass without critical errors affecting the dashboard or models.

## Dashboard Startup
- [ ] Run `streamlit run src/dashboard.py`
- [ ] Dashboard opens successfully in the browser.
- [ ] No immediate Python exceptions on startup.

## Dashboard Sections Validation
- [ ] **Overview Tab**: Data from `results/summary.jsonl` loads successfully. Global metrics are visible.
- [ ] **Client/Heterogeneity Tab**: Per-client metrics bar chart renders correctly.
- [ ] **FL Training Tab**: Training curves load successfully when selecting an experiment (e.g., `exp3_fedavg`).
- [ ] **Robustness Tab**: Comparison chart between FedAvg, Poisoned FedAvg, and FedMedian loads and displays the correct drops/recoveries.
- [ ] **Explainability Tab**: SHAP model and test samples load correctly from `results/exp10_defense/`.
- [ ] **Transaction Explanation**: Moving the transaction slider updates the fraud probability and human-readable explanation successfully.

## Required Files Verification
- [ ] `src/dashboard.py` exists and is functional.
- [ ] `results/summary.jsonl` exists and contains data for all 10 experiments.
- [ ] `results/exp10_defense/model.pt` exists.
- [ ] `results/exp10_defense/scaler.pkl` exists.
- [ ] `results/exp10_defense/test_sample_X.npy` exists.
- [ ] `results/exp10_defense/test_sample_y.npy` exists.
- [ ] `PROJECT_PROGRESS.md` is updated and reflects demo-readiness.

## Backup Assets
- [ ] Pre-captured screenshots of the dashboard are available (Recommended).
- [ ] Final report generation script (`scratch/generate_final_report.py`) runs successfully as a fallback.
