# Final Project Structure

```text
.
├── configs/             # YAML configurations defining FL hyperparameters, neural network sizes, and data generation seeds
├── docs/                # Project documentation, checklists, runbooks, and product guides
├── results/             # Validated outputs, JSONL summary metrics, and frozen PyTorch checkpoints
├── src/                 # Main source code directory
│   ├── app/             # Streamlit UI logic, page layouts, and session state
│   ├── experiments/     # Reproducible experiment configurations (Exp 1 through 10)
│   ├── federated/       # Flower (flwr) simulation logic, customized strategies, and client definitions
│   └── services/        # Backend abstraction layer mapping FL models to UI components
├── tests/               # Comprehensive Pytest test suite ensuring validation parity
├── README.md            # Final product overview and launch instructions
└── requirements.txt     # Locked project dependencies
```

## Core Directories

- **`src/app/`**: Contains the code responsible for the interactive Streamlit dashboard. It separates frontend logic from the backend machine learning model.
- **`src/services/`**: Exposes clear APIs (`ResultsService`, `ModelService`, `ExplainabilityService`) for the frontend to consume frozen models and output metadata safely, without regenerating data.
- **`src/federated/`**: Contains the core logic required to run the `flwr` based simulated federated learning loop, including robust implementations like FedMedian.
- **`src/experiments/`**: Scripts detailing specific variations of FL runs (e.g., class imbalance, non-IID splits, poisoning attacks).
- **`results/`**: The most critical directory for the final application. It houses the historical records of training metrics and the final saved model parameters required for the dashboard inference engine.
