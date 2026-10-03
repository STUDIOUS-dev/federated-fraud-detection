# Privacy-Aware & Explainable Federated Fraud Detection

## Overview
I built this project to study how federated learning (FL) can be used for fraud detection across heterogeneous banks. My goal was to see if multiple independent financial institutions could collaboratively train a machine learning model while keeping their raw customer transaction data local. 

## Problem
Detecting financial fraud requires large, diverse datasets. However, banks may face privacy, security, governance, and competitive constraints when sharing raw customer transaction data. This creates isolated "data silos," limiting the performance of any individual bank's fraud detection system. I wanted to explore how federated fraud detection could bridge this gap by allowing banks to learn from each other's fraud patterns while preserving data privacy.

## Key Features
I implemented and tested several core features throughout this project:
- **Heterogeneous bank simulation**: I generated synthetic transaction data simulating four different banking profiles (e.g., retail-heavy vs. high-value transactions).
- **Centralized & local baselines**: I built baseline PyTorch models to compare FL against a centralized reference baseline and isolated silos.
- **Federated algorithms**: I integrated Flower (`flwr`) to experiment with `FedAvg`, `FedProx`, and `FedMedian` aggregation strategies.
- **Poisoning experiments**: I introduced a malicious client to study model collapse and robustness under data-poisoning attacks.
- **SHAP explainability**: I added SHAP to provide human-readable, localized feature attribution for individual transactions.
- **Streamlit dashboard**: I created an interactive web application to visualize the frozen experiments and dynamically analyze transactions.

## Architecture
```text
+---------------------------+
|      Global Server        |
| FedAvg / FedProx /        |
| FedMedian                 |
+-------------+-------------+
              |
      +-------+-------+-------+-------+
      |       |       |       |       |
   Bank A  Bank B  Bank C  Bank D
   Local   Local   Local   Local
   Model   Model   Model   Model
      |       |       |       |
 Local Data Local Data Local Data Local Data
```
For the backend architecture, I used PyTorch to build a Tabular Multilayer Perceptron (MLP) and orchestrated the federated learning simulation using `flwr` (Flower). I then built a Streamlit application that acts as an interactive dashboard, allowing users to explore the final frozen models and evaluation results.

## Technology Stack
- **Python**: Core logic and scripting
- **PyTorch**: Deep learning backend
- **Flower (`flwr`)**: Federated learning orchestration
- **SHAP**: Explainability and feature attribution
- **Streamlit**: Web dashboard interface
- **Scikit-Learn / Pandas / NumPy**: Data processing and metric evaluation
- **Pytest**: Automated testing framework

## Project Structure
Here is how I structured the final repository:
```text
.
├── configs/             # Configuration files for FL rounds, architecture, etc.
├── data/                # Data generation output (local-only/ignored)
├── docs/                # Project documentation and guides
├── results/             # Validated outputs, final metrics, and frozen checkpoints
├── scratch/             # Scratch scripts (local-only/ignored)
├── src/
│   ├── app/             # Streamlit dashboard pages and state
│   ├── experiments/     # Reproducible FL scripts (Exp 1-10)
│   ├── federated/       # Flower clients and custom strategies (FedAvg, FedProx, FedMedian)
│   └── services/        # Abstraction layer mapping core FL to UI components
├── tests/               # Pytest suite
├── README.md
└── requirements.txt
```

## Installation

### Windows
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
# If flwr[simulation] requires Ray and you face dependency issues with protobuf:
pip install "ray[default]"
```

### Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the Product
To launch the interactive demonstration dashboard that I built:
```bash
streamlit run src/dashboard.py
```

## Run Tests
To validate the current integrity of the codebase:
```bash
python -m pytest -v
```

## Research Experiments
I have already run all the core research experiments, and their final metrics and model checkpoints are stored in the `results/` directory. The Streamlit dashboard reads directly from these frozen artifacts. 
If you want to fully reproduce one of my experiments (for example, the FedProx run), you can execute:
```bash
python -m src.experiments.exp4_fedprox
```

## Final Results
These are the benchmark results I obtained after running the baseline and federated configurations:

| Configuration | PR-AUC |
|---|---:|
| Centralized | 0.2508 |
| FedAvg | 0.2439 |
| FedProx | 0.2456 |
| IID | 0.2120 |
| Unequal Sizes | 0.4077 |
| Class Imbalance | 0.3089 |
| Dropout | 0.2478 |
| Poisoned FedAvg | 0.0036 |
| Poisoned FedMedian | 0.2435 |

I observed that clean FedAvg performed closely to the centralized reference baseline. However, when I introduced a malicious client (Poisoned FedAvg), the model collapsed. Switching the aggregation strategy to FedMedian substantially reduced the impact of the poisoned client under the tested configuration.

## Explainability
I integrated SHAP (SHapley Additive exPlanations) to help interpret the model's predictions. The SHAP module provides localized feature attribution, which allows the dashboard to generate human-readable explanations of which transaction features contributed most to the model's fraud probability score. The dashboard generates these explanations without requiring access to the original raw training dataset during inference.

## Limitations
While building this project, I made several scoping decisions. The current limitations are:
- **Synthetic/controlled data**: The data distribution I used is synthesized, not actual production banking data.
- **Simulated banks**: The network consists of locally simulated clients rather than physically distributed nodes over a real network.
- **No Differential Privacy**: I did not implement DP bounding on the weight updates.
- **No Secure Aggregation**: Cryptographic masking of model updates is absent. Without Secure Aggregation, model updates are not cryptographically protected against inference attacks such as gradient inversion.
- **Limited attack coverage**: I only tested the model's robustness against one specific data-poisoning vector.
- **No real banking deployment**: This is an offline research simulation, not a production-ready application.
- **No real network latency/bandwidth evaluation**: Communication overhead was not measured during the simulation.

## Future Work
If I were to expand on this project, I would focus on:
- Implementing **Differential Privacy** to bound information leakage from weight updates.
- Adding **Secure Aggregation** using cryptographic masking.
- Scaling the simulation to more clients and evaluating actual communication/latency costs.
- Testing the architecture against stronger, coordinated Byzantine attacks.
- Validating the approach against an external, real-world dataset.
- Implementing communication-efficiency analysis.
