# Final Product Plan: Privacy-Aware & Explainable Federated Fraud Detection

## Product Architecture
The final product will be a coherent Streamlit-based application layered on top of the validated Federated Learning (FL) and Machine Learning (ML) core. It will function as a **simulation and demonstration system**, conceptually organized into two main layers:

1. **Application Layer**: User Interface (Streamlit), application state, and decoupled services (model service, result service, explainability service).
2. **Core ML/FL Layer**: Existing data generation, feature engineering, FL simulation, and model definition modules.

The application will read from the already frozen research results (`results/summary.jsonl` and artifact folders) rather than dynamically running long FL training loops during standard usage.

## Application Components
1. **Streamlit UI**: Multi-page/tab application for navigation and visual presentation.
2. **Result Service**: Abstraction to safely load and parse `summary.jsonl`, per-client history, and configurations.
3. **Model Service**: Abstraction to load the pre-trained final model checkpoint (from `exp10_defense`) and run inference.
4. **Explainability Service**: Reusable SHAP explainer service to attribute model predictions to features without directly tying to UI code.
5. **Core ML Modules**: The existing PyTorch tabular MLP, preprocessing, and causal feature engineering.

## Data Flow
- **Historical Results**: Read-only flow from `results/` -> Result Service -> Streamlit UI.
- **Model Inference**: Read-only test transactions (`results/.../test_sample_X.npy`) -> Model Service -> Prediction -> Explainability Service (SHAP) -> Streamlit UI.
- **FL Training (Optional/Demonstration)**: Small-scale config -> Core FL Framework -> Ephemeral Result -> UI (only on explicit user request).

## UI Structure
The Streamlit application will have clear navigation (e.g., sidebar or tabs) mapping to the required views:
1. **Home**: Project overview, architecture diagram, objective.
2. **Bank Simulation**: Data distribution, heterogeneity details for Banks A, B, C, and D.
3. **Federated Training**: UI to inspect the FL process (FedAvg, FedProx).
4. **Model Performance**: Global vs. per-client metric visualizations (PR-AUC, F1, etc.).
5. **Robustness**: Visualizing the verified results of Clean FedAvg vs Poisoned FedAvg vs FedMedian.
6. **Transaction Analyzer**: Interactive tool to pick a test transaction and view predictions.
7. **Explainability**: SHAP attribution and human-readable explanation for the selected transaction.
8. **Project Information**: Methodology, limitations, and future work.

## Backend/service structure
The backend logic will be refactored into domain-specific services:
- `src/services/results.py`: Functions to load and parse the final experiment outputs.
- `src/services/model.py`: Model loading and transaction prediction abstractions.
- `src/services/explainability.py`: Wrapper for SHAP (building on existing `src/explainability.py`).

## Reusable existing modules
- `src/model.py` (FraudMLP)
- `src/preprocessing.py` & `src/feature_engineering.py` (Data handling)
- `src/evaluation.py` (Metrics)
- `src/explainability.py` (Core SHAP logic)
- `results/` directory (Authoritative frozen results)
- `src/config.py` (Configuration dataclasses)

## New modules required
- `src/app/` (Streamlit pages, state management, and main UI)
- `src/services/results.py`
- `src/services/model.py`
- Integration code to connect the services to the UI pages.

## Configuration strategy
- The existing `src/config.py` and `configs/default.yaml` will remain the source of truth for ML/FL parameters.
- Application-specific configurations (like file paths for models and data, UI defaults) will be added either to the UI logic directly or a dedicated UI config class. 
- UI actions will not silently mutate or overwrite the core research configurations.

## Model lifecycle
- **Loading**: The UI will load the model strictly via the Model Service using pre-saved checkpoints (e.g., `results/exp10_defense/model.pt`).
- **Error Handling**: Graceful degradation if checkpoints are missing (clear user message instead of a crash).
- **Execution**: Run purely in `eval()` mode.

## Result loading strategy
- **Summary**: Read from `results/summary.jsonl` via Result Service.
- **Detailed History**: Load `results.json` from specific experiment folders when deep-diving into rounds.
- **No mutations**: The application will only read these files and never write or overwrite historical data.

## Error handling
- Missing result files or checkpoints will trigger `st.warning` or `st.error` rather than Python tracebacks.
- Invalid test samples will be caught by the Model Service before entering the PyTorch inference graph.
- Malformed JSONL records will be safely skipped.

## Testing strategy
- Maintain the current 19/19 passing test suite.
- Add tests for new services (`test_results_service.py`, `test_model_service.py`).
- Add tests for UI initialization and missing artifact handling to prevent runtime dashboard crashes.

## Final folder structure
```text
src/
├── app/                  # New UI module
│   ├── pages/            # Streamlit multi-page structures
│   ├── components/       # Reusable UI widgets
│   └── state/            # Session state management
├── services/             # New backend service layer
│   ├── results.py
│   ├── model.py
│   └── explainability.py # Migrated/wrapped from current explainability.py
├── federated/            # Existing FL logic
├── experiments/          # Existing experiment framework
├── model.py              # Existing PyTorch model
├── preprocessing.py      # Existing data logic
├── feature_engineering.py# Existing feature logic
├── evaluation.py         # Existing metrics
├── config.py             # Existing config
└── dashboard.py          # Entry point for the Streamlit app
```

## Definition of Done
1. **Architecture**: Application logic cleanly separated from Streamlit UI components (services layer created).
2. **UI Implementation**: All specified views (Home, Bank Simulation, Training, Performance, Robustness, Analyzer, Explainability, Info) are fully functional.
3. **Integrity**: Frozen research results are correctly loaded and displayed without modification.
4. **Resilience**: The application handles missing artifacts and malformed data gracefully.
5. **Testing**: `pytest` passes 100% of the augmented test suite.
6. **Documentation**: `README.md`, `docs/FINAL_PRODUCT_GUIDE.md`, and `PROJECT_PROGRESS.md` are accurately updated.
