# Project Progress

## Project Status
v1.1 - research-validity revision (Phases A-C complete). Product and documentation complete; research conclusions revised after multi-seed validation (see `docs/STUDY_RESULTS.md`).

### Research Status
Complete for the scope below. Headline results are 5-seed means; the ten original experiments are single-seed (seed 42) demonstrations.

### Product Status
Complete

### Documentation Status
Complete (reconciled with the multi-seed study on 2026-11-08)

### Demo Status
Ready

## Project Objective
Simulate multiple independent banks collaboratively training a fraud-detection model without sharing raw customer transaction data. Investigate how effectively FL improves fraud detection across heterogeneous financial institutions and its robustness under realistic conditions.

## Current Architecture
- Synthetic transaction generation (`data_generation.py`), causal features (`feature_engineering.py`), temporal split (`preprocessing.py`)
- PyTorch tabular MLP (`model.py`), metrics (`evaluation.py`: PR-AUC, ROC-AUC, recall at 1% FPR, F1/precision/recall at 0.5)
- Flower federated layer (`src/federated/`): clients (honest + 3 attack types), tracked strategies (FedAvg, FedProx via client term, FedMedian, FedTrimmedAvg, Krum) with seeded client dropout
- Experiment framework (`src/experiments/framework.py`): shared reference test pool for all scenarios, fully seeded training, Flower/Ray runner and a Ray-free in-process runner
- Multi-seed study (`src/experiments/study.py`, `study_report.py`) -> `results/study/`, `docs/STUDY_RESULTS.md`
- Services + Streamlit app (`src/services/`, `src/app/`, `src/dashboard.py`); helper scripts in `scripts/`

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
Results: superseded. Early figures came from unseeded single runs on a different test pool; see Final Results and `docs/STUDY_RESULTS.md`.
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
- Superseded by the multi-seed study: FedAvg and FedProx (mu=0.01) are indistinguishable (PR-AUC 0.300 +/- 0.069 for both).
- Config: 4 clients, 5 rounds, 2 local epochs, defaults
Issues: `flwr[simulation]` required installing `ray`, which had dependency resolution conflicts with `protobuf`. Fixed via `--ignore-installed` and isolating Ray installation.

### Milestone 3: Advanced FL Experiments (Heterogeneity & Sizing)
Status: Completed
Objective: Implement configurable IID / Non-IID splits and client sizing (Exp 5 & 6) and class imbalance/dropout/malicious experiments (Exp 7-9).
Completed: All 5 experiments completed with unified test framework.
Results: superseded by the multi-seed study (`docs/STUDY_RESULTS.md`). Earlier single-run readings (e.g. 'Exp 9 total collapse', 'small banks suffer') did not hold up across seeds or used non-comparable test pools.

### Milestone 4: Advanced Scenarios & Explainability
Status: Completed
Objective: Robust defence (FedMedian), SHAP explainability, Streamlit dashboard. Delivered 2026-09-15.

## Module Status

| Module | Status | Tests | Notes |
|---|---|---|---|
| `data_generation` | Complete | Yes | Synthetic banks; IID profile = mean of bank profiles |
| `feature_engineering` | Complete | Yes | Causal features |
| `preprocessing` | Complete | Yes | Scaling, temporal split |
| `model` | Complete | Yes | PyTorch MLP |
| `federated` | Complete | Yes | Flower clients/strategies, attacks, seeded dropout |
| `evaluation` | Complete | Yes | Adds ROC-AUC, recall@1% FPR |
| `experiments` (framework, study) | Complete | Yes | Shared test pool, in-process runner, multi-seed study |
| services / app | Complete | Yes | Dashboard reads frozen results and study aggregate |

## Experiments

Single-seed (seed 42) results of the ten original experiments are in `results/<exp>/` and `results/summary.jsonl`. Use `docs/STUDY_RESULTS.md` (5 seeds) for any comparison.

| Priority | Experiment | Status | Notes |
|---|---|---|---|
| Must | 1. Centralized baseline | Completed | `results/exp1_centralized` |
| Must | 2. Local-only models | Completed | `results/exp2_local` |
| Must | 3. FedAvg | Completed | `results/exp3_fedavg` |
| Must | 4. FedProx (mu=0.01) | Completed | `results/exp4_fedprox`; mu sweep 0.01/0.1/1.0 in the study |
| Must | 5. IID vs Non-IID | Completed | `results/exp5_iid` |
| Must | 6. Unequal client sizes | Completed | `results/exp6_sizes` |
| Must | 7. Class imbalance | Completed | `results/exp7_imbalance` |
| Should | 8. Client dropout | Completed | Now genuine per-round dropout (p=0.5); previously random client sampling |
| Optional | 9. Malicious client (weight scaling, bank_d) | Completed | `results/exp9_malicious` |
| Optional | 10. FedMedian defence (same attacker) | Completed | `results/exp10_defense`; previously used a different attacker than Exp 9 |
| Added | 11. Multi-seed study | Completed | 38 scenarios x 5 seeds: sweeps, 3 attacks, 4 aggregators, 1-2 attackers |

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
See Final Quantitative Results below and `docs/STUDY_RESULTS.md`.

## Technical Decisions
- Preserve existing working behavior before refactoring.
- PyTorch MLP is the primary model.
- FedProx acts as a client-side proximal term, server uses standard weighted aggregation.
- Evaluation must support global + per-client metrics.
- Reusable experiment framework to track configs and reproducible results.
- Class-weighted loss for imbalance.
- Strict temporal causality in features.

## Research Conclusions (revised 2026-11-08, 5-seed study)
1. **Did federation improve over local-only training?** Slightly. +0.008 PR-AUC (5/5 seeds) but no gain in recall at a fixed 1% false-positive rate. It matches pooled-data training (-0.009, not significant). The earlier 'FL generalized better than local' claim rested on one seed.
2. **Does it help more for weak banks?** Yes: unequal sizes +0.032 PR-AUC, extreme imbalance +0.069 PR-AUC over local training (the 10-customer bank goes from 0.04 to 0.31).
3. **IID vs strong heterogeneity / FedProx?** No measurable difference; the synthetic banks are only mildly heterogeneous. FedProx mu <= 0.1 equals FedAvg; mu=1.0 is slightly worse (not significant).
4. **Did unequal sizes disadvantage small banks?** Not supported: small banks gained relative to training alone.
5. **Dropout?** No measurable effect at 25% or 50% with 5 rounds.
6. **Poisoning of FedAvg?** Weight scaling by -10 from one of four clients hurts badly but bimodally across seeds (0.098 +/- 0.130). Sign-flip and label-flip were ineffective at these settings.
7. **Defences?** FedMedian, FedTrimmedAvg and Krum all neutralise one attacker at no measurable cost. With two of four attackers median and trimmed mean fail; Krum survives only under favourable, non-adaptive conditions.
8. **Explainability?** SHAP gives model-attribution explanations of individual predictions (not causal).
9. **Limitations:** synthetic data with ~53 test fraud cases per seed, no DP / secure aggregation, simple non-adaptive attacks, absolute accuracy modest (F1 ~0.04 at threshold 0.5).

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
- Optional extensions (Phase E): differential privacy, secure aggregation, communication cost, real or public benchmark data, adaptive attackers.

## Demo Readiness
**Status:** READY

**Validated Commands:**
- Test Suite: `python -m pytest -v`
- Dashboard: `streamlit run src/dashboard.py` (Validated in headless mode)
- Fallback Report: `python scripts/generate_final_report.py`

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
- 2026-11-08: Phases A-C (research-validity revision).
  - A: reconciled README, report, progress file, runbook, release notes with measured results; `scratch/` -> tracked `scripts/`.
  - B: fully seeded training and initialisation (repeat runs bit-identical); shared reference test pool for all scenarios; 5-seed study with mean +/- std and paired comparisons; Ray-free in-process runner verified equal to Flower/Ray.
  - C: real client dropout; IID defined as mean bank profile; FedProx mu sweep; sign-flip / label-flip attacks and multi-attacker support; FedTrimmedAvg and Krum; ROC-AUC and recall@1% FPR; exp10 now uses the same attacker as exp9; exp4 now really runs mu=0.01; case-insensitive aggregation names.
  - Re-ran exp1-exp10 (single-seed numbers changed vs v1.0 because of the shared test pool and seeding).
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

### Headline: multi-seed study
Mean +/- std over 5 seeds, shared reference test pool: see `docs/STUDY_RESULTS.md`. Key rows (PR-AUC): Centralized 0.309 +/- 0.075, Local 0.292 +/- 0.070, FedAvg 0.300 +/- 0.069, FedProx 0.300 +/- 0.069; FedAvg with one scaling attacker 0.098 +/- 0.130, FedMedian with the same attacker 0.300 +/- 0.070.

### Reproducibility Check (exp5_iid)
Run 1 PR-AUC: 0.24650818546984377
Run 2 PR-AUC: 0.24650818546984377 (identical: training is fully seeded)

### Single-seed demonstration runs (seed 42, `results/summary.jsonl`)
Noise between seeds is larger than most differences in this table; do not rank rows by it.

| Setup | PR-AUC | Recall@1%FPR | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Centralized | 0.2525 | 0.3208 | 0.0383 | 0.0200 | 0.4528 |
| Local Average | 0.2388 | 0.3113 | 0.0305 | 0.0160 | 0.5236 |
| FedAvg | 0.2480 | 0.3019 | 0.0388 | 0.0201 | 0.5283 |
| FedProx (mu=0.01) | 0.2483 | 0.3019 | 0.0387 | 0.0201 | 0.5283 |
| IID | 0.2465 | 0.3208 | 0.0529 | 0.0280 | 0.4717 |
| Unequal Sizes | 0.2423 | 0.3019 | 0.0934 | 0.0537 | 0.3585 |
| Class Imbalance | 0.2474 | 0.3019 | 0.0472 | 0.0249 | 0.4717 |
| Dropout (p=0.5) | 0.2471 | 0.3019 | 0.0289 | 0.0148 | 0.5283 |
| Poisoned FedAvg (scale x-10) | 0.0064 | 0.0566 | 0.0000 | nan | 0.0000 |
| Poisoned FedMedian | 0.2498 | 0.3208 | 0.0451 | 0.0236 | 0.5094 |

## Final Product Validation

- **Page-by-page validation**: Completed successfully using Streamlit AppTest. All pages initialize correctly without exceptions, and optional missing data is handled gracefully.
- **Transaction Analyzer validation**: Completed. Selected a held-out test transaction, applied the expected 14-dimensional feature schema, loaded the frozen model, successfully executed inference, and retrieved the target label.
- **SHAP validation**: Completed. Verified that SHAP uses exactly the same input model and produces correctly dimensioned attribution matching the model prediction (model attribution, not causal). Human-readable explanation generation works.
- **Robustness validation**: Completed. Robustness page now reads values from results/summary.jsonl (no hard-coded fallbacks) and shows the multi-seed table.
- **Documentation validation**: Completed. All required documentation files (README, FINAL_PRODUCT_GUIDE, DEMO_RUNBOOK, DEMO_CHECKLIST, FINAL_PRODUCT_PLAN) are present and describe the current architecture.
- **Final test count**: 48/48 tests passing (100%) after the 2026-11-08 revision.
- **Final status**: FINAL PRODUCT READY

