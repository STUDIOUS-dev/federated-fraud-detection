# Release Notes

## v1.2.0 (Phase E: communication, privacy, adaptive attacks)

### Added
- Per-round communication accounting (bytes up/down per client) in every strategy.
- Update-norm clipping (`clip_norm`) and client-level differential privacy (`dp_noise_multiplier`, `dp_delta`) with a zCDP ε accountant, applied server-side in `aggregate_fit` (`src/federated/privacy_robustness.py`).
- Adaptive colluding attacks `alie` (with `alie_z`) and `minmax`, crafted from the round's honest updates.
- Phase E study: 53 scenarios × 5 seeds (`python -m src.experiments.study_e`, `--workers N`), results in `results/study_e/` and `docs/PHASE_E_RESULTS.md`.
- Dashboard page "Privacy & Attacks".
- Tests for clipping, DP noise and accounting, adaptive attacks, byte counting, and Flower/in-process equality under DP + dropout and ALIE + clipping.

### Findings
- About 38 KiB of traffic per round for four banks; 5 rounds reach full accuracy.
- Clipping costs ≤ 0.009 PR-AUC and neutralises the scaling attacker (0.102 → 0.298).
- Client-level DP: ε ≈ 13 gives 0.224 (worse than local-only training); ε ≤ 6 is unusable with four banks.
- ALIE with 2 of 4 attackers (z = 5) breaks Krum on 2/5 seeds and FedMedian/FedTrimmedAvg on 1/5; FedAvg is unaffected. Weaker settings (z ≤ 3, Min-Max) had no measurable effect except on Krum.

### Compatibility
- All new options default to off; v1.1 results (`results/exp*`, `results/study/`) are unchanged and were not regenerated.

### Still not covered
- Secure aggregation; DP at a usable scale; Phase E on the OpenML data; optimisation-based adaptive attacks.

---

## v1.1.0

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
