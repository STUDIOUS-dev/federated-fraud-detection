# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Simulation of federated fraud detection across four heterogeneous synthetic banks (PyTorch MLP + Flower `flwr` simulation, SHAP explainability, Streamlit dashboard). The project is marked "FINAL PRODUCT — FROZEN" (see `PROJECT_PROGRESS.md`): `results/` holds frozen experiment outputs that the dashboard reads, so avoid regenerating or overwriting them casually.

## Commands

Run everything from the repo root (imports are `src.*`, results paths are relative to cwd).

```bash
pip install -r requirements.txt          # flwr simulation needs Ray; if protobuf conflicts: pip install "ray[default]"
python -m pytest -v                      # full suite
python -m pytest tests/test_federated.py::test_flower_client_serialization -v   # single test
streamlit run src/dashboard.py           # dashboard (adds repo root to sys.path itself)
uvicorn src.api:app --port 8000          # scoring API: GET /health, POST /score {transactions:[14 features], threshold, explain}
python -m src.experiments.exp4_fedprox   # run one experiment (exp1..exp10); writes to results/<exp_name>/
```

```bash
python scripts/run_all_experiments.py                  # regenerate exp1-10 (Flower+Ray, ~20 min; overwrites results/)
python -m src.experiments.study [--only REGEX]         # 38 scenarios x 5 seeds, resumable; writes results/study/ + docs/STUDY_RESULTS.md
python -m src.experiments.study --report-only          # rebuild the report from results/study/runs.jsonl
```

No linter is configured. `scripts/` holds helper/acceptance scripts. `tests/test_ui.py` uses a 3 s AppTest timeout, so it can fail on a cold start if run alone first; run it after `tests/test_dashboard.py` or the full suite.

## Architecture

**Data pipeline** (`src/`): `data_generation.generate_federation` builds per-bank synthetic transactions from `BANK_PROFILES` in `config.py` → `feature_engineering.add_causal_features` (causal only, no target leakage; `BASE_FEATURES` defines model inputs) → `preprocessing.temporal_split` (split by timestamp, not random) → `fit_scaler` on pooled train data → `to_xy`.

**Config**: `configs/default.yaml` is loaded by `config.load_config()` into the flat `Config` dataclass. Experiments copy the default config and override fields via `setattr` (e.g. `client_sizes`, `dropout_rate`, `malicious_client_index`, `aggregation`, `heterogeneity`).

**Experiment framework** (`src/experiments/framework.py`): `ExperimentFramework` seeds RNGs, dumps config, prepares per-client + global test data, and `save_results` writes `results/<exp>/results.json` and appends to `results/summary.jsonl`. `run_fl_experiment(name, overrides)` wires Flower: `federated.client.get_client_fn` + `federated.server.get_strategy`, runs `fl.simulation.start_simulation` (Ray), and saves `model.pt` and `test_sample_{X,y}.npy` (used by SHAP). `build_data` always scores every scenario on the same reference test pool (default heterogeneity/sizes, same seed). `run_fl_inprocess` is a Ray-free runner using the same clients/strategies (verified bit-identical to Flower), used by `study.py`; `runners.py` has the centralized/local baselines. Each `expN_*.py` is a thin wrapper calling this with overrides; exp1/exp2 are non-federated baselines.

**Federated layer** (`src/federated/`): `client.FraudClient` (Flower NumPyClient; local training in `local_training.py` with per-client/round seeding, FedProx proximal term, `MaliciousFraudClient` with `cfg.attack` = scale | signflip | labelflip), `server.py` has `Tracked*` strategies (FedAvg/FedProx/FedMedian/FedTrimmedAvg/Krum, names case-insensitive, seeded `dropout_rate`) that record per-client metrics and participating clients per round and keep `latest_params` for checkpointing.

**App layer**: `src/dashboard.py` (Streamlit entry) + `src/app/` (pages, state) → `src/services/` (`results.py`, `model.py`, `explainability.py`) which read frozen `results/` artifacts. `src/explainability.py` holds the SHAP logic.

Note: `PROJECT_PROGRESS.md` mentions legacy `federated.py`/`run_baseline.py`; those are now the `src/federated/` package. `docs/` has demo runbooks and the final report.
