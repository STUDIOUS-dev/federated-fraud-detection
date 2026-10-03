# Release Notes

### Release
`v1.0.0`

### Product
Privacy-Aware & Explainable Federated Fraud Detection

### Included
- FL simulation
- FedAvg
- FedProx
- FedMedian
- robustness experiments
- SHAP
- Streamlit application
- reproducible results
- 29 tests

### Known limitations
- The data is synthetic/controlled data, not real banking data.
- The network contains locally simulated banks.
- No Differential Privacy is implemented, allowing potential gradient inversion risks.
- No Secure Aggregation is implemented.
- Evaluated against a limited attack coverage (single vector poisoning).
- No real banking deployment or real network latency/bandwidth evaluations.

### Validation
All 29 automated Pytest validations passed successfully (100%), alongside verified AppTest automated UI assertions. All model loading, preprocessing, inference mapping, and SHAP attributions operate identically on frozen checkpoints. 

*(Note: This project is an academic simulation and not intended for immediate production readiness.)*
