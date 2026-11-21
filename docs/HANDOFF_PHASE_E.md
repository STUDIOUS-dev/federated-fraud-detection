# Handoff: Phase E (v1.2) — status and how to continue

Written 2026-11-21 for whoever picks this up next (a person or an AI agent).
Branch: `claude/ecstatic-mayer-fbjo9k` (2 commits ahead of `main`, no PR opened yet).

## 1. Goal

v1.1 ended with these optional "Phase E" items: communication cost, differential privacy (DP),
secure aggregation, adaptive attackers, real data. The plan was to do **E0** (housekeeping), **E1**
(communication cost), **E2** (clipping + client-level DP) and **E4** (adaptive attacks), plus a dashboard
page, then update the docs and release v1.2. **E3** (secure aggregation) was dropped on purpose: in this
simulation it would only demonstrate the protocol, and it cannot be combined with FedMedian or Krum.
**E5** (repeat E2/E4 on the OpenML data) is optional and was blocked here because the network policy
denies openml.org.

## 2. Done (committed and pushed)

| Item | Where | Status |
|---|---|---|
| Clipping, DP noise, zCDP ε, ALIE + Min-Max attacks, byte counting | `src/federated/privacy_robustness.py` | done, tested |
| Strategy integration (attack → clip → DP noise in `aggregate_fit`; `communication_history`, `update_norm_history`) | `src/federated/server.py` | done, tested |
| New config fields `clip_norm`, `dp_noise_multiplier`, `dp_delta`, `alie_z`; attacks `alie`/`minmax` | `src/config.py`, `configs/default.yaml` | done; all default to off |
| Malicious client trains honestly for adaptive attacks | `src/federated/client.py` | done |
| Study rows now include `communication` (+ `privacy` when DP is on) | `src/experiments/study.py` (`_phase_e_fields`) | done |
| Phase E suite, 45 scenarios × 5 seeds, `--workers N` | `src/experiments/study_e.py`, `study_e_report.py` | done |
| Results (225 runs, complete) | `results/study_e/runs.jsonl`, `aggregate.json` | done |
| Generated tables (data only, no interpretation) | `docs/PHASE_E_RESULTS.md` | done |
| Dashboard page "Privacy & Attacks" | `src/app/pages/privacy_attacks.py`, `src/dashboard.py` | done |
| Tests | `tests/test_phase_e.py` (17), `tests/test_pages.py` (+1), `tests/test_inprocess.py` (+2 Ray cases) | passing |
| CLAUDE.md command + architecture notes | `CLAUDE.md` | done |

Verified:
- With the defaults, results are **bit-identical** to before the change (FedAvg seed 42 gives the same PR-AUC).
- The Flower/Ray simulation and the in-process runner agree exactly, including under DP + dropout and
  ALIE + clipping: `RUN_RAY_TESTS=1 FL_NUM_CPUS=2 python -m pytest tests/test_inprocess.py` gave 4 passed
  (needs `pip install "flwr[simulation]"`).
- `--workers 4` (single-threaded torch per run) produces rows identical to the serial runner.
- Full suite: 66 passed before Phase E. The new tests passed on their own; the **full suite has not been
  re-run since the results landed** (see step 3 below).

## 3. Still to do (in order)

1. **Optional: stronger ALIE setting.** At z = 1 and z = 3 the adaptive attacks hardly hurt anything (see §4).
   A one-seed check at z = 5 with 2 attackers broke FedMedian and Krum (PR-AUC about 0.01). To show where the
   attacks start to work, add `"alie_z5": {"attack": "alie", "alie_z": 5.0}` to `ADAPTIVE` in
   `src/experiments/study_e.py`, add `"alie_z5"` to the attack loops in `study_e_report.py`
   (`comparisons()` and the section-5 table) and `src/app/pages/privacy_attacks.py` (`attack_table`), then run
   `python -m src.experiments.study_e --workers 4` (resumable, so only the new runs execute; about 5 minutes).
2. **Write the interpretation** (the generated `docs/PHASE_E_RESULTS.md` holds numbers only):
   - `docs/FINAL_REPORT.md`: add section "7.4 Phase E: communication, privacy, adaptive attacks"; revise §8
     Conclusion and §8.1 Future Work (DP, adaptive attacks and comm cost are now done; secure aggregation and real-data replication remain).
   - `README.md`: add a Phase E bullet to Key Features, a Phase E results table and command under Research
     Experiments / Final Results, and update **Limitations** (the "No Differential Privacy" and
     "non-adaptive attacks" bullets are no longer accurate) and **Future Work**.
   - `docs/RELEASE_NOTES.md`: add a `v1.2.0` section.
   - `PROJECT_PROGRESS.md`: status v1.2, Phase E milestone, change-log entry, and fix the stale test count
     ("48/48"; it was 66 before Phase E and is now higher; take the number from step 3).
3. **Run the full suite**: `python -m pytest -q`. The dashboard page test now renders the new page with real data.
4. Commit, push, and open a PR to `main` (only if the owner asks for one).
5. Optional E5: run E2/E4 on the OpenML creditcard data. `src/experiments/real_data_study.py` has its own
   aggregation loop and does not use the strategies yet, so it needs refactoring first. It also needs network
   access to openml.org.

## 4. Findings so far (5 seeds, PR-AUC mean ± std; clean FedAvg = 0.301 ± 0.069)

These are the inputs for the write-up. The full tables and paired tests are in `docs/PHASE_E_RESULTS.md`.

- **Baselines reproduce v1.1 in substance.** Fed vs local +0.005 (5/5 seeds, t = 6.2); fed vs central −0.005
  (n.s.). Numbers differ from `docs/STUDY_RESULTS.md` only in the 3rd–4th decimal because of newer
  torch/numpy versions, which is why Phase E re-ran its own baselines.
- **Communication (E1).** Traffic is about 38 KiB per round for all 4 banks (up + down; the model has
  1,025 parameters). PR-AUC by number of rounds: 1 round 0.284, 2 rounds 0.295, 3 rounds 0.297,
  5 rounds 0.301 (190 KiB), 10 rounds 0.300. Five rounds of updates move less data than pooling the
  training features once (about 3.2 MB).
- **Clipping alone (E2)** costs little: C = 1.0 gives 0.300, 0.5 gives 0.297, 0.25 gives 0.292.
- **Client-level DP (E2) is very costly with only 4 clients.**

  | ε | PR-AUC |
  |---|---|
  | ≈ 31 (z = 0.5) | 0.285 |
  | ≈ 13 (z = 1) | 0.224 |
  | ≈ 6 (z = 2) | 0.041 |
  | ≈ 2.8 (z = 4) | 0.004 |

  DP at ε ≈ 13 is worse than training locally without federation (−0.072, 0/5 seeds). Ten rounds at z = 1
  (ε ≈ 20) did not help (0.200). The explanation is that the noise per bank is too large when only 4 banks
  are averaged; real DP-FL deployments use hundreds of clients.
- **Clipping is a strong, cheap defence against the v1.1 scaling attacker.** FedAvg under attack scores
  0.102 ± 0.148; with clipping at 0.5 it scores 0.298 (+0.196, 5/5 seeds, t = 4.0), which is not
  distinguishable from clean. DP at z = 1 under the same attack gives 0.196.
- **Adaptive attacks (E4) were weak at these strengths.** ALIE (z = 1, 3) and Min-Max, with 1 or 2 of 4
  attackers, leave FedAvg, FedMedian and FedTrimmedAvg at about 0.30. The only visible damage is Krum
  under ALIE z = 3 with 2 attackers: 0.242 ± 0.145 (high variance; −0.059 vs clean, n.s.). Clipping did not
  clearly help Krum there (0.276 ± 0.085, n.s.). A likely reason: small learning rate, 2 local epochs and
  5 rounds give small, tightly clustered honest updates, which leaves little room for stealthy attacks.
  This is the reason for step 3.1.

## 5. Key commands

```bash
pip install -r requirements.txt
python -m pytest -q
python -m src.experiments.study_e --workers 4          # resumable; skips finished (scenario, seed) pairs
python -m src.experiments.study_e --only alie_z5 --workers 4
python -m src.experiments.study_e --report-only        # rebuild docs/PHASE_E_RESULTS.md + aggregate.json
streamlit run src/dashboard.py                         # page "Privacy & Attacks"
RUN_RAY_TESTS=1 FL_NUM_CPUS=2 python -m pytest tests/test_inprocess.py   # needs flwr[simulation]
```

## 6. Rules to keep

- Do **not** regenerate or overwrite the frozen v1.1 outputs (`results/exp*`, `results/study/`,
  `results/summary.jsonl`, `docs/STUDY_RESULTS.md`). Phase E writes only to `results/study_e/` and
  `docs/PHASE_E_RESULTS.md`.
- New options must default to off so the v1.1 results stay reproducible. `tests/test_phase_e.py::test_default_strategy_is_unchanged_by_phase_e`
  guards this.
- Report differences as paired comparisons across seeds; with 5 seeds, |t| > 2.8 is roughly the 5% level.
- Threat model for the adaptive attacks: the attackers see all honest updates and collude. That is the
  worst case, and the docs should say so.
