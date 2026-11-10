# Release Notes

### Release
`v1.1.0` (research-validity revision of v1.0.0)

### Product
Privacy-Aware & Explainable Federated Fraud Detection

### Included
- FL simulation: FedAvg, FedProx, FedMedian, FedTrimmedAvg, Krum
- Three poisoning attacks (weight scaling, sign-flip, label-flip), 1-2 attackers
- Genuine client dropout; IID / unequal-size / imbalance scenarios
- Fully seeded, bit-reproducible training
- Multi-seed study (38 scenarios x 5 seeds) with paired comparisons: `docs/STUDY_RESULTS.md`
- SHAP explainability and Streamlit dashboard (robustness page now shows multi-seed results)
- Tests (see PROJECT_PROGRESS.md for the current count)

### Changes from v1.0.0
- All scenarios are scored on one shared reference test pool; single-seed numbers from v1.0.0 are not comparable and were regenerated.
- Corrected claims: federation vs local is a small gain (+0.008 PR-AUC), large only for weak banks; dropout, IID-ness and FedProx showed no measurable effect; the poisoning collapse is seed-dependent.
- Fixed: exp10 used a different attacker than exp9; exp4 ran mu=0.1 while documented as 0.01; `dropout_rate` was unused; `scratch/` helper scripts were untracked (now `scripts/`).

### Known limitations
- Synthetic data with a weak fraud signal; ~53 fraud cases per test pool, so per-bank metrics are very noisy.
- Locally simulated banks; no network latency/bandwidth evaluation.
- No Differential Privacy or Secure Aggregation (gradient-inversion risk).
- Simple non-adaptive attacks; Krum result with 2 of 4 attackers is outside its theoretical guarantee.
- Absolute accuracy is modest (PR-AUC ~0.30, F1 ~0.04 at 0.5).

*(Academic simulation; not intended for production.)*
