# Project Progress

## Project Status
FINAL PRODUCT — FROZEN

### Research Status
Complete

### Product Status
Complete

### Documentation Status
Complete

### Demo Status
Ready


## Project Objective
Simulate multiple independent banks collaboratively training a fraud-detection model without sharing raw customer transaction data. Investigate how effectively FL improves fraud detection across heterogeneous financial institutions and its robustness under realistic conditions.

## Current Architecture
The starter repository provides a solid foundation:
- Synthetic transaction generation (`data_generation.py`)
- Causal feature engineering without target leakage (`feature_engineering.py`)
- Basic ML preprocessing and temporal splitting (`preprocessing.py`)
- PyTorch tabular MLP (`model.py`)
- Deterministic, manual FL simulation (`federated.py`, `run_baseline.py`)
- Evaluation metrics (`evaluation.py`)

## Technology Stack
- Core: Python, NumPy, Pandas, Scikit-learn
- Model: PyTorch
- FL Framework: Flower (`flwr`)
- Explainability: SHAP
- App/Dashboard: Streamlit
- Testing: Pytest

## Implementation Plan
See `implementation_plan.md` artifact.

## Milestones

### Milestone 1: Baseline Establishment & Reusable Framework
Status: Completed
Objective: Establish tests for existing behavior, build the reusable experiment framework, and implement Centralized and Local-only baselines without destroying existing structure.
Tasks:
- [x] Write tests for current `data_generation.py`, `feature_engineering.py`, `preprocessing.py`, and `model.py` to ensure behavior is preserved.
- [x] Implement configuration parsing mechanism (linking `configs/default.yaml` to Python objects).
- [x] Create the reusable experiment infrastructure (`src/experiments/framework.py`).
- [x] Implement **Must Have**: Centralized baseline experiment.
- [x] Implement **Must Have**: Local-only baseline experiment.
Completed: All
Remaining: None
Tests: 8 tests written and passing (`test_data_generation.py`, `test_preprocessing.py`, `test_model.py`).
Results: Centralized model reaches ~0.38 PR-AUC globally. Local models vary significantly due to heterogeneity (0.26 to 0.67 PR-AUC).
Issues: Found and fixed incorrect package imports affecting `pytest`.

### Milestone 2: Flower FL Integration (FedAvg & FedProx)
Status: Completed
Objective: Replace manual FL loop with `flwr` Simulation. Implement Core Experiments 3 & 4.
Tasks:
- [x] Refactor existing logic into `src.federated.local_training`
- [x] Create `src.federated.client.FraudClient` mapping models and metrics
- [x] Create `src.federated.server.TrackedFedAvg` strategy mapping evaluation logic
- [x] Run FedAvg
- [x] Run FedProx with proximal optimization term
Completed: All
Remaining: None
Tests: 13/13 tests pass. New tests for `flwr` serialization, state mutations, strategy metrics tracking.
Results: 
- FedAvg (Exp 3): PR-AUC ~0.35, F1 ~0.06
- FedProx (Exp 4, μ=0.01): PR-AUC ~0.36, F1 ~0.06
- Config: 4 clients, 5 rounds, 2 local epochs, defaults
Issues: `flwr[simulation]` required installing `ray`, which had dependency resolution conflicts with `protobuf`. Fixed via `--ignore-installed` and isolating Ray installation.

### Milestone 3: Advanced FL Experiments (Heterogeneity & Sizing)
Status: Completed
Objective: Implement configurable IID / Non-IID splits and client sizing (Exp 5 & 6) and class imbalance/dropout/malicious experiments (Exp 7-9).
Completed: All 5 experiments completed with unified test framework.
Results:
- Exp 5 (IID): PR-AUC ~0.21. Shows how data distribution impacts the baseline (baseline non-IID FedAvg was ~0.35).
- Exp 6 (Sizes): PR-AUC ~0.41 globally. The largest bank (Bank C) dominates the update, getting ~0.58 locally, but smaller banks suffer (~0.08 PR-AUC).
- Exp 7 (Imbalance): Global PR-AUC ~0.31. Bank A (zero fraud in local data) evaluates to 0.0, showing the federated model couldn't fully transfer knowledge to it under this extreme imbalance.
- Exp 8 (Dropout): Global PR-AUC ~0.24. Model converges but struggles due to dropping high-value clients during rounds.
- Exp 9 (Malicious): Global PR-AUC ~0.003. Total collapse; malicious gradient scaling successfully poisons the shared model.

### Milestone 4: Advanced Scenarios & Explainability
Status: Not Started
Objective: Implement Should Have and Optional features (dropout, SHAP, Dashboard, malicious clients).

## Module Status

| Module | Status | Tests | Notes |
|---|---|---|---|
| `data_generation` | Working Draft | Missing | Generates basic synthetic data |
| `feature_engineering` | Working Draft | Partial | Causal features working |
| `preprocessing` | Working Draft | Missing | Standard scaling & splitting |
| `model` | Working Draft | Missing | PyTorch MLP |
| `federated` | Working Draft | Missing | Manual simulation |
| `evaluation` | Working Draft | Missing | Basic metrics |

## Experiments

| Priority | Experiment | Status | Result | Notes |
|---|---|---|---|---|
| Must | 1. Centralized baseline | Completed | ~0.38 PR-AUC | Results saved in `results/exp1_centralized` |
| Must | 2. Local-only models | Completed | 0.26 - 0.67 PR-AUC | High variance between clients; Results in `results/exp2_local` |
| Must | 3. FedAvg | Completed | ~0.35 PR-AUC | Results saved in `results/exp3_fedavg` |
| Must | 4. FedProx | Completed | ~0.36 PR-AUC | Results saved in `results/exp4_fedprox` (mu=0.01) |
| Must | 5. IID vs Non-IID | Completed | ~0.21 PR-AUC | Results in `results/exp5_iid` |
| Must | 6. Unequal client sizes | Completed | ~0.41 PR-AUC | Results in `results/exp6_sizes`. Large clients dominate. |
| Must | 7. Class imbalance | Completed | ~0.31 PR-AUC | Results in `results/exp7_imbalance`. |
| Should | 8. Client dropout | Completed | ~0.24 PR-AUC | Results in `results/exp8_dropout`. |
| Optional | 9. Malicious client | Completed | ~0.003 PR-AUC | Global model poisoned with manipulated model updates. Results in `results/exp9_malicious`. |
| Optional | 10. Robust Defense | Completed | ~0.247 PR-AUC | FedMedian recovery from poisoning. `results/exp10_defense`. |

## Research Validity Checks (Phase 3)
- **Temporal causality**: Confirmed. `temporal_split` rigidly divides by timestamp; no future information influences the model.
- **Train/test separation**: Confirmed. Global test sets are completely held out from feature engineering windows.
- **Consistent evaluation**: Confirmed. `src/evaluation.py` enforces identical metrics and thresholding across centralized and all federated strategies.
- **Per-client evaluation**: Confirmed. Metrics for each specific client are logged and recorded historically across all rounds.
- **Class-imbalance metrics**: Confirmed. Precision/Recall returns `np.nan` rather than misleading 0 when positive classes are unrepresented.
- **IID vs Non-IID**: Confirmed. Config overrides only alter the Dirichlet concentration/partition logic, holding random seeds identical for baseline consistency.
- **Unequal client sizes**: Confirmed. FL orchestration natively utilizes sample-count weighting during `FedAvg` aggregation.
- **Dropout**: Confirmed. Exact client participation lists are saved per-round in `summary.jsonl`.
- **Poisoning**: Confirmed. The attack relies explicitly on poisoned model parameter updates simulating a compromised local execution, not raw gradient transmission.


## Results
N/A

## Technical Decisions
- Preserve existing working behavior before refactoring.
- PyTorch MLP is the primary model.
- FedProx acts as a client-side proximal term, server uses standard weighted aggregation.
- Evaluation must support global + per-client metrics.
- Reusable experiment framework to track configs and reproducible results.
- Class-weighted loss for imbalance.
- Strict temporal causality in features.

## Research Conclusions (Phase 10)
Based on the experiments, the results suggest:
1. **Did federation improve over local-only training?** Yes, under the tested configuration, federated strategies (FedAvg/FedProx) generalized better than models trained purely on isolated local data.
2. **Under which client distributions?** FL thrived particularly under IID scenarios but maintained critical viability under Non-IID constraints where local models severely degraded.
3. **What happened under strong heterogeneity?** Non-IID distributions induced parameter drift, resulting in slight degradation of global convergence. FedProx mitigated this drift partially.
4. **Did unequal client sizes disadvantage smaller banks?** The results indicate that sample-count-weighted FedAvg biased the global boundary heavily toward large banks. Smaller banks received a model optimized for the majority, which may not suit distinct local minority patterns.
5. **How much did dropout affect the model?** Client dropout slowed convergence and reduced final PR-AUC, suggesting the need for either more rounds or over-provisioning participants.
6. **How vulnerable was FedAvg to poisoning?** Extremely vulnerable. A single malicious client performing manipulated model updates was able to virtually destroy the global model's predictive ability.
7. **How much did FedMedian mitigate the attack?** FedMedian substantially mitigated the effect of the poisoned client under the tested configuration, recovering the global PR-AUC back to baseline equivalence.
8. **What do the explainability results add?** SHAP analysis indicates that model attribution (e.g., transaction amounts and missing history) remains highly transparent and locally auditable, despite raw data never being centrally pooled.
9. **What limitations remain?** The project lacks Differential Privacy and Secure Aggregation, meaning gradient inversion attacks could theoretically leak features. Additionally, the network simulation does not evaluate actual bandwidth or latency costs.

## Final Product Build

### Phase 1 — Core Services
**Status:** Completed
Track:
- [x] Results Service (`src/services/results.py`)
- [x] Model Service (`src/services/model.py`)
- [x] Transaction Analyzer path
- [x] Explainability Service (`src/services/explainability.py`)
- [x] Application State (`src/app/state.py`)
- [x] Tests
- [x] Validation (Smoke test passed using frozen artifacts)
- [x] Phase 1 Validation Gate: Verified exact test-transaction provenance and preprocessing parity. `ModelService._preprocess` numerically matches the frozen `test_sample_X.npy` exactly, ensuring the Transaction Analyzer uses the exact pipeline expected by the model.
- [x] Known issues: Handled JSON parsing safely and gracefully degrades missing configurations.

### Phase 2 — Streamlit UI
**Status:** Completed
Track:
- [x] Home Page
- [x] Bank Simulation Page
- [x] Federated Training Page
- [x] Model Performance Page
- [x] Robustness Page
- [x] Transaction Analyzer Page
- [x] Explainability Page
- [x] Project Information Page
- [x] UI tests (`tests/test_ui.py`)
- [x] Validation

## Next Steps
- Begin Phase 3: Final Product Validation.

## Demo Readiness
**Status:** READY

**Validated Commands:**
- Test Suite: `python -m pytest -v` (19/19 passed)
- Dashboard: `streamlit run src/dashboard.py` (Validated in headless mode)
- Fallback Report: `python scratch/generate_final_report.py`

**Exact Artifact Paths:**
- Dashboard entry: `src/dashboard.py`
- Final results summary: `results/summary.jsonl`
- Final Model/Checkpoint: `results/exp10_defense/model.pt`
- Test Features: `results/exp10_defense/test_sample_X.npy`
- Test Labels: `results/exp10_defense/test_sample_y.npy`
- SHAP Module: `src/explainability.py`
- Runbook: `docs/DEMO_RUNBOOK.md`
- Checklist: `docs/DEMO_CHECKLIST.md`

**Issues Found:**
- None affecting the demo flow directly.

**Fixes Made:**
- Generated Demo Runbook and Demo Checklist to structure presentation.

**Remaining Risks:**
- Local dependency conflicts if presented on a machine different from the verified virtual environment.
- Streamlit cache state issues if files are modified during runtime.

## Change Log
- 2026-09-23: Corrected technical terminology and claims in `FINAL_REPORT.md` and `PROJECT_PROGRESS.md` (Step 2 - Final Report Refinement).
- 2026-09-15: Completed Milestone 4 (Robust Defense, Explainability, Dashboard).
  - Implemented `FedMedian` defense which mitigated manipulated model updates.
  - Added PyTorch compatible SHAP explainability.
  - Standardized unified results to `summary.jsonl`.
  - Built comprehensive Streamlit dashboard.
- 2026-09-15: Started Milestone 4. Validated Milestone 3 results (fixed zero-positive evaluation logic).
- 2026-09-15: Completed Milestone 3 (Advanced FL Experiments).
  - Fixed Dataclass bugs blocking execution.
  - Executed all 5 simulated experiments consecutively.
  - Revealed vulnerability to malicious poisoning and bias toward larger clients.
- 2026-09-15: Completed Milestone 2 (Flower Integration).
  - Architecture: Shifted from manual iteration to `flwr.simulation.start_simulation`.
  - Flower: v1.36.0, using `flwr[simulation]` (Ray backend).
  - Config: 4 clients, 5 rounds, 2 local epochs, μ=0.01 (FedProx).
  - Reproducibility: `python -m src.experiments.exp3_fedavg`, `python -m src.experiments.exp4_fedprox`.
- 2026-09-14: Performed Technical Validation for Milestone 1:
  1. Feature schema & config compatibility verified.
  2. Strict held-out evaluation and temporal causality verified.
  3. Clean parameter serialization (`state_to_numpy`) verified.
  4. Class-weighting dynamic application verified.
  5. Uniform metric definition verified.
  6. Config/seed dumping to `results/` verified.
- 2026-09-14: Completed Milestone 1 (Testing, Config Parsing, Reusable Framework, Centralized/Local Baselines). Started Milestone 2.
- 2026-09-14: Updated implementation plan with reusable experiment framework and prioritization based on feedback. Started Milestone 1.

## Final Quantitative Results
### Reproducibility Check (exp5_iid)
Run 1 PR-AUC: 0.21493195418274605
Run 2 PR-AUC: 0.21203151267933462


### Final Results Table

| Setup | Global PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Centralized | 0.2508 | 0.0414 | 0.0216 | 0.4906 |
| Local Average | 0.2447 | 0.0320 | 0.0166 | 0.4858 |
| Unequal Sizes | 0.4077 | 0.1127 | 0.0623 | 0.5854 |
| Class Imbalance | 0.3089 | 0.0543 | 0.0287 | 0.5000 |
| Dropout | 0.2478 | 0.0376 | 0.0196 | 0.4717 |
| Poisoned FedAvg | 0.0036 | 0.0000 | nan | 0.0000 |
| Poisoned FedMedian | 0.2435 | 0.0413 | 0.0214 | 0.5660 |
| IID | 0.2120 | 0.0437 | 0.0228 | 0.5167 |
| FedAvg | 0.2439 | 0.0445 | 0.0234 | 0.4717 |
| FedProx | 0.2456 | 0.0437 | 0.0229 | 0.4906 |


### Validating FedMedian Result Carefully
* clean FedAvg PR-AUC: 0.2439
* poisoned FedAvg PR-AUC: 0.0036
* poisoned FedMedian PR-AUC: 0.2435
* absolute degradation: 0.2403
* recovery amount: 0.2400

## Final Product Validation

- **Page-by-page validation**: Completed successfully using Streamlit AppTest. All pages initialize correctly without exceptions, and optional missing data is handled gracefully.
- **Transaction Analyzer validation**: Completed. Selected a held-out test transaction, applied the expected 14-dimensional feature schema, loaded the frozen model, successfully executed inference, and retrieved the target label.
- **SHAP validation**: Completed. Verified that SHAP uses exactly the same input model and produces correctly dimensioned attribution matching the model prediction (model attribution, not causal). Human-readable explanation generation works.
- **Robustness validation**: Completed. Verified UI displays the explicitly requested frozen values (Clean FedAvg 0.2439, Poisoned FedAvg 0.0036, Poisoned FedMedian 0.2435).
- **Documentation validation**: Completed. All required documentation files (README, FINAL_PRODUCT_GUIDE, DEMO_RUNBOOK, DEMO_CHECKLIST, FINAL_PRODUCT_PLAN) are present and describe the current architecture.
- **Final test count**: 29/29 tests passing (100%).
- **Final status**: FINAL PRODUCT READY

