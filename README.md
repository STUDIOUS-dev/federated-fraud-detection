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
├── scripts/             # Helper scripts: run all experiments, acceptance/smoke tests, report generator
├── src/
│   ├── app/             # Streamlit dashboard pages and state
│   ├── experiments/     # Reproducible FL scripts (Exp 1-10, multi-seed `study.py`)
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
The original ten experiments (`src/experiments/exp1_*` ... `exp10_*`) are single-seed (seed 42) demonstrations; their outputs and model checkpoints are in `results/` and are what the dashboard shows. To regenerate all of them (takes ~20 minutes; Flower + Ray):
```bash
python scripts/run_all_experiments.py
python -m src.experiments.exp4_fedprox     # or one at a time
```

Because single seeds are noisy (the test pool has only ~53 fraud cases), the headline results come from a **multi-seed study**: 38 scenarios x 5 seeds, every scenario scored on the same reference test pool. It uses an in-process runner that drives the same clients and strategies as Flower (and reproduces the Flower/Ray result exactly), so it finishes in about an hour instead of many.
```bash
python -m src.experiments.study                 # resumable; writes results/study/ and docs/STUDY_RESULTS.md
python -m src.experiments.study --only attack   # a subset (regex on scenario name)
python -m src.experiments.study --report-only   # rebuild the report from results/study/runs.jsonl
```

## Final Results
Global PR-AUC, mean ± std over 5 seeds (full tables, per-bank results and paired tests: [docs/STUDY_RESULTS.md](docs/STUDY_RESULTS.md)):

| Configuration | PR-AUC | Recall @ 1% FPR |
|---|---:|---:|
| Centralized | 0.309 ± 0.075 | 0.378 ± 0.064 |
| Local-only (avg) | 0.292 ± 0.070 | 0.374 ± 0.073 |
| FedAvg | 0.300 ± 0.069 | 0.379 ± 0.076 |
| FedProx (mu=0.01) | 0.300 ± 0.069 | 0.379 ± 0.076 |
| FedAvg, 1 of 4 clients scaling weights x -10 | 0.098 ± 0.130 | |
| FedMedian, same attacker | 0.300 ± 0.070 | |

What the data supports:
- Federation matches pooling the data and beats isolated training by a small, consistent margin (+0.008 PR-AUC, 5/5 seeds); the gain is large for banks with little data or fraud (+0.07 PR-AUC in the extreme-imbalance scenario).
- IID vs non-IID data, FedProx (mu up to 0.1) and client dropout (up to 50%) made no measurable difference in this synthetic setting.
- FedAvg is clearly vulnerable to a weight-scaling attacker; FedMedian, trimmed mean and Krum all neutralise one attacker of four. With two of four malicious, median and trimmed mean fail; Krum survives only under favourable assumptions (see the caveats in the study report).
- Absolute accuracy is modest (F1 about 0.04 at threshold 0.5) because the synthetic fraud signal is weak.

## Explainability
SHAP provides localized feature attribution, which allows the dashboard to generate human-readable explanations of which transaction features contributed most to the model's fraud probability score. These are model attributions, not causal explanations.

## Limitations
- **Synthetic data** with a weak fraud signal and about 53 fraud cases in each test pool; per-bank results are very noisy.
- **Simulated banks** on one machine; no real network latency or bandwidth measured.
- **No Differential Privacy** and **no Secure Aggregation**; model updates could leak information (e.g. gradient inversion).
- **Limited attacks:** three simple, non-adaptive poisoning attacks, at most 2 of 4 clients.
- Only mild heterogeneity between banks, so null results for IID-ness and FedProx may not carry over to real data.
- Not a production system.

## Future Work
Differential privacy and secure aggregation; adaptive and colluding attackers; real or public benchmark data; communication-cost analysis.
