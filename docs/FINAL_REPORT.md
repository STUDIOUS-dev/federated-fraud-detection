# 1. Title Page

**Project Title:** Privacy-Aware & Explainable Federated Fraud Detection
**Student Name:** [Student Name]
**Degree/Program:** [Degree/Program]
**University:** [University]
**Academic Session:** [Academic Session]
**Project Guide:** [Guide Name]

---

# 2. Abstract

Financial fraud remains a massive challenge, but the highly sensitive nature of banking data restricts cross-institutional data sharing. Federated Learning (FL) offers a decentralized approach to train global models without exposing raw local datasets. In this project, I simulated a financial network of four heterogeneous synthetic banks and trained a PyTorch Multilayer Perceptron (MLP) with FedAvg and FedProx, comparing against centralized and local-only baselines. All headline numbers are means over 5 seeds scored on one shared reference test pool (about 53 fraud cases per seed), because single-seed results proved too noisy to interpret (centralized PR-AUC ranged from 0.25 to 0.44 across seeds). Under this setup federated training matched pooled-data training (PR-AUC 0.300 ± 0.069 vs 0.309 ± 0.075) and beat isolated local training by a small but consistent margin (+0.008 PR-AUC in 5 of 5 seeds), with a much larger benefit for banks with little data or little fraud (+0.07 PR-AUC in the extreme-imbalance scenario). FedProx, client dropout and IID versus non-IID data produced no measurable difference in this synthetic setting. A single malicious client sending weights scaled by -10 degraded FedAvg sharply (0.098 ± 0.130 PR-AUC, bimodal across seeds), while FedMedian, FedTrimmedAvg and Krum all stayed at about 0.30; with two of four clients malicious, median and trimmed-mean failed (about 0.003) and only Krum survived. I also integrated SHAP for model-attribution explanations of individual predictions. The study has important limits: synthetic data, no differential privacy or secure aggregation, and attacks that are not adaptive.

---

# 3. Introduction

The detection of fraudulent financial transactions is an ongoing arms race between financial institutions and malicious actors. Traditionally, banks rely on institution-specific models trained solely on their internal data. While this approach protects customer privacy, it severely limits the model's exposure to diverse fraud patterns, leaving smaller banks particularly vulnerable. Centralizing data across multiple banks to train a unified model could theoretically solve this limitation, but it is often practically impossible due to strict data privacy regulations, competitive boundaries, and the massive risks associated with pooling sensitive financial records.

Federated Learning (FL) emerges as a decentralized alternative, allowing multiple institutions to collaboratively train a shared model without ever transmitting raw transaction data. Instead of data, banks exchange model parameters/updates with a central server. Raw transaction data remains local to the simulated client, though it is important to note that I did not implement Differential Privacy and Secure Aggregation in this baseline. The server aggregates these updates to form a global model. I built this project to explore the feasibility and robustness of FL in a simulated financial network, focusing on how data heterogeneity and malicious participants affect the global model's performance, and how robust aggregation strategies and explainability frameworks can address these challenges.

---

# 4. Problem Statement

Individual financial institutions lack the diverse, cross-institutional data required to train highly generalized fraud detection models, yet they cannot share raw customer data due to privacy constraints. Existing federated solutions theoretically solve this data-sharing challenge but often struggle with the extreme statistical heterogeneity typical of real-world banking (e.g., varying client sizes and severe class imbalances) and are vulnerable to malicious participants attempting to poison the shared model.

---

# 5. Research Question

How effectively can Federated Learning improve fraud detection across heterogeneous financial institutions and what is its robustness under realistic conditions (such as data imbalance, unequal client sizes, and adversarial poisoning attacks)?

---

# 6. Methodology

## 6.1 Data Generation and Feature Engineering
To simulate a realistic environment, I generated synthetic, temporally sequenced financial transactions for four independent banks (Bank A, B, C, and D). To prevent data leakage, I implemented strict temporal splitting and engineered causal features (e.g., historical transaction counts and averages) that only look backward in time.

## 6.2 Federated Architecture
The system I built employs a server-client architecture using the Flower (`flwr`) framework. The underlying fraud detection model is a PyTorch-based Multilayer Perceptron (MLP). Training is fully seeded (data, initial weights, shuffling, dropout, client sampling) so repeated runs are bit-identical. Each bank acts as a local client that trains the model on its proprietary dataset. I evaluated five aggregation strategies:
* **FedAvg:** Averages the local model weights, weighted by the number of samples at each client.
* **FedProx:** Introduces a proximal term to the local loss function to prevent local models from drifting too far from the global model, addressing non-IID data distributions.
* **FedMedian:** A robust aggregation strategy that takes the coordinate-wise median of the local model updates instead of the mean, reducing the influence of extreme or manipulated model updates.
* **FedTrimmedAvg:** Drops the largest and smallest value of each coordinate (one client per side with four clients) and averages the rest.
* **Krum:** Selects the single client update closest to its nearest neighbours, discarding the rest.

## 6.3 Explainability
To address the "black-box" nature of neural networks, I integrated SHapley Additive exPlanations (SHAP). The SHAP module provides model-attribution explanations for individual transaction predictions made by the global model on local client data. It highlights the features (e.g., transaction amount, time since last transaction) driving the fraud probability, supporting interpretability and auditability of individual model predictions, but it is not a causal explanation or proof of regulatory compliance.

---

# 7. Experiments and Results

All numbers below are mean ± std over 5 seeds (42-46) and are scored on one shared reference test pool per seed, so every scenario is comparable. The complete tables, per-bank results and paired comparisons are in `docs/STUDY_RESULTS.md` (generated by `python -m src.experiments.study`). The ten original experiments (`results/exp1`-`exp10`) are single-seed (seed 42) demonstrations used by the dashboard and should not be used for comparisons.

## 7.1 Baselines and core federated performance

| Setup | PR-AUC | Recall at 1% false-positive rate |
|---|---|---|
| Centralized (pooled data) | 0.309 ± 0.075 | 0.378 ± 0.064 |
| Local-only (average of 4 banks) | 0.292 ± 0.070 | 0.374 ± 0.073 |
| FedAvg | 0.300 ± 0.069 | 0.379 ± 0.076 |
| FedProx (mu = 0.01 / 0.1 / 1.0) | 0.300 / 0.300 / 0.287 | 0.379 / 0.374 / 0.375 |

*Finding:* Federation is statistically indistinguishable from pooling the data (paired difference -0.009 PR-AUC, FedAvg wins 1 of 5 seeds). Federation beats isolated training by a small, consistent PR-AUC margin (+0.008, 5 of 5 seeds), but the recall at a fixed alert budget does not differ (+0.005, within noise). FedProx gave no benefit over FedAvg at any tested mu; mu = 1.0 was slightly worse but not significantly. In this dataset, a single bank's data is already close to sufficient, so most of the achievable accuracy is reachable without federation. The earlier claim that "federated strategies generalize better than isolated local models" was based on one seed and is only weakly supported.

## 7.2 Impact of heterogeneity, size, imbalance and dropout

* **IID vs non-IID training data:** no measurable difference (-0.004 PR-AUC, not significant). The synthetic bank profiles are only mildly heterogeneous, so FedProx and IID-ness have little to fix.
* **Unequal client sizes** (200 / 50 / 1000 / 750 customers): federation beat local training by +0.032 PR-AUC and +0.034 recall at 1% FPR (4-5 of 5 seeds). The smallest bank benefits most (bank_b PR-AUC 0.156 local vs 0.238 federated). The earlier claim that size-weighted FedAvg "left smaller banks behind" is not supported: they gained relative to training alone.
* **Extreme imbalance** (bank_a reduced to 10 customers): federation beat local training by +0.069 PR-AUC (5 of 5 seeds). A bank with almost no data gets a usable model only through federation (bank_a local 0.041 vs 0.311 federated).
* **Client dropout** (25% and 50% of clients missing each round): no measurable effect with 5 rounds. The earlier claim that dropout slows convergence was an artefact of a single seed.

Per-bank results are noisy: bank_a and bank_d have very few fraud cases in their own test sets (std often larger than the mean).

## 7.3 Vulnerability and robust defences

PR-AUC, mean ± std over 5 seeds; one attacker is bank_d.

| Attack | FedAvg | FedMedian | FedTrimmedAvg | Krum |
|---|---|---|---|---|
| none | 0.300 ± 0.069 | 0.299 ± 0.070 | 0.299 ± 0.070 | 0.297 ± 0.071 |
| weight scaling x -10 (1 of 4) | 0.098 ± 0.130 | 0.300 ± 0.070 | 0.300 ± 0.070 | 0.294 ± 0.069 |
| sign-flip x5 (1 of 4) | 0.294 ± 0.077 | 0.299 ± 0.071 | 0.299 ± 0.071 | 0.294 ± 0.069 |
| label-flip (1 of 4) | 0.300 ± 0.071 | 0.298 ± 0.069 | 0.298 ± 0.069 | 0.294 ± 0.069 |
| weight scaling (2 of 4) | 0.004 ± 0.001 | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.289 ± 0.072 |
| sign-flip (2 of 4) | 0.003 ± 0.001 | 0.003 ± 0.001 | 0.003 ± 0.001 | 0.289 ± 0.072 |

* The weight-scaling attack is the only one that clearly harms FedAvg, and its effect is bimodal across seeds (the single-seed demo, seed 42, happens to be near the collapse case). Sign-flip and label-flip are weak at these training settings (small local updates), so they say little about resilience.
* Robust aggregators cost nothing measurable when no attacker is present.
* With 50% attackers, coordinate-wise median and trimmed mean fail as theory predicts. Krum survives here, but it returns a single client's model, is outside its n > 2f+2 guarantee at n = 4, and the attackers were non-adaptive; colluding or mimicking attackers would probably defeat it.

## 7.4 Phase E: communication cost, privacy and adaptive attacks

A follow-up study (`python -m src.experiments.study_e`; 53 scenarios × 5 seeds, same reference test pool; tables and paired tests in `docs/PHASE_E_RESULTS.md`) added the extensions that v1.1 listed as future work. Baselines were re-run in that study's environment (FedAvg 0.301 ± 0.069, local 0.295, central 0.306) so that every comparison below is paired.

**Communication cost.** The model has 1,025 parameters, so one round moves about 38 KiB across all four banks (upload and download). Five rounds (190 KiB) reach the full FedAvg accuracy; one round already gives 0.284 and ten rounds give no further gain (0.300). For comparison, pooling the training features centrally would move about 3.2 MB of customer-level data once. Bandwidth is not a constraint for a model of this size.

**Update clipping and client-level differential privacy.** Clipping each bank's update to an L2 norm of 1.0, 0.5 or 0.25 costs at most 0.009 PR-AUC (not significant). Adding Gaussian noise for client-level DP (each bank's whole contribution is protected; ε from zCDP composition, δ = 1e-5, no amplification claimed) is very costly with only four banks:

| Noise multiplier z | ε after 5 rounds | PR-AUC |
|---|---|---|
| 0 (clip 0.5 only) | ∞ | 0.297 ± 0.070 |
| 0.5 | 31.5 | 0.285 ± 0.074 |
| 1.0 | 13.2 | 0.224 ± 0.035 |
| 2.0 | 6.0 | 0.041 ± 0.034 |
| 4.0 | 2.8 | 0.004 ± 0.001 |

At ε ≈ 13 the federated model is already worse than training locally without federation (−0.072, 0 of 5 seeds), and at ε ≤ 6 it is useless. Training for ten rounds at z = 1 (ε ≈ 20) did not recover accuracy (0.200). The noise is added to an average over only four updates, so its standard deviation per bank is large compared with the signal; practical DP-FL relies on hundreds or thousands of clients. **In a four-bank consortium, client-level DP with a meaningful ε is not practical**; record-level DP or secure aggregation would have to provide the privacy instead.

**Clipping as a defence.** Against the v1.1 scaling attacker (1 of 4), clipping updates to 0.5 restores FedAvg from 0.102 ± 0.148 to 0.298 ± 0.074 (+0.196, 5 of 5 seeds, t = 4.0), indistinguishable from the clean model. It is as effective as switching to a robust aggregator and costs almost nothing. DP noise on top (z = 1) gives 0.196, because the noise costs more than the attack.

**Adaptive attacks.** Two attacks designed to evade robust aggregation were added: ALIE (each attacker sends mean − z·std of the honest updates per coordinate) and Min-Max (the largest shift that stays no farther from any honest update than the honest updates are from each other). The threat model is the worst case: attackers see all honest updates every round and collude.

PR-AUC, 2 colluding attackers (bank_c, bank_d) of 4:

| Attack | FedAvg | FedMedian | FedTrimmedAvg | Krum |
|---|---|---|---|---|
| none (only FedAvg was re-run) | 0.301 ± 0.069 | – | – | – |
| ALIE z = 1 | 0.300 ± 0.068 | 0.301 ± 0.069 | 0.301 ± 0.069 | 0.301 ± 0.069 |
| ALIE z = 3 | 0.301 ± 0.069 | 0.305 ± 0.069 | 0.305 ± 0.069 | 0.242 ± 0.145 |
| ALIE z = 5 | 0.303 ± 0.070 | 0.242 ± 0.145 | 0.242 ± 0.145 | 0.164 ± 0.162 |
| Min-Max | 0.300 ± 0.069 | 0.301 ± 0.069 | 0.301 ± 0.069 | 0.301 ± 0.069 |

* With a single attacker, no adaptive attack changed any aggregator's PR-AUC by more than 0.005.
* With two attackers, the damage is concentrated in the robust aggregators, not in FedAvg. At z = 5, FedMedian and FedTrimmedAvg collapse on 1 of 5 seeds (PR-AUC 0.008) and Krum on 2 of 5 (0.004 and 0.008), with a third Krum seed degraded (0.167 vs 0.254). Plain FedAvg is unaffected (0.303), because averaging two shifted updates with two honest ones halves the shift. This is the reverse of the scaling attack and the opposite of the intuition that "robust" aggregators are always safer: an attacker who knows the defence can target it, and Krum, which selects a single update, is the easiest to steer.
* The effects are seed-dependent (bimodal), so the paired differences are not significant at 5 seeds (Krum at z = 5: −0.137 ± 0.135, t = −2.3). They should be read as "can break" rather than "breaks on average".
* Clipping in front of the aggregator did not clearly help against ALIE (Krum at z = 3: +0.034, n.s.), because ALIE updates have normal-sized norms by design.
* The attacks are weak at z ≤ 3 because honest updates are small and tightly clustered (low learning rate, 2 local epochs, 5 rounds), which leaves little room for a stealthy shift. Stronger, optimisation-based attacks were not tried.

## 7.5 Phase E on real data (OpenML credit-card)

The Phase E scenarios were repeated on the public OpenML credit-card dataset (`python -m src.experiments.real_data_e`; 24 scenarios × 5 seeds; tables in `docs/REAL_DATA_PHASE_E_RESULTS.md`). The data and the four banks are those of the earlier real-data check (temporal 80/20 split, banks holding 45 / 30 / 15 / 10% of the training rows with an unequal fraud share, 10 rounds), but training now uses the same clients and strategies as the synthetic study, so clipping, DP and the adaptive attacks apply unchanged. The test split has only 75 frauds.

| Setting (PR-AUC, 5 seeds) | Real data | Synthetic (§7.4) |
|---|---|---|
| FedAvg / local-only / centralized | 0.807 / 0.791 / 0.793 | 0.301 / 0.295 / 0.306 |
| FedAvg, updates clipped | 0.805 (C = 1.0) | 0.297 (C = 0.5) |
| Client-level DP, z = 0.5 / 1.0 / 2.0 | 0.668 / 0.066 / 0.005 (ε ≈ 50 / 20 / 9) | 0.285 / 0.224 / 0.041 (ε ≈ 31 / 13 / 6) |
| 1 scaling attacker: FedAvg / + clipping / FedMedian | 0.300 / 0.797 / 0.802 | 0.102 / 0.298 / 0.300 |
| 2 ALIE z = 5 attackers: FedAvg / FedMedian / Krum | 0.803 / 0.791 / 0.406 ± 0.368 | 0.303 / 0.242 / 0.164 |

* **Baselines** reproduce the earlier real-data check (FedAvg 0.807 vs 0.803, FedMedian 0.808 vs 0.806). Federation beats isolated training more clearly than on synthetic data (+0.016, 5 of 5 seeds, t = 3.7); the difference from centralized training is within noise.
* **Communication** is about 530 KiB in total over 10 rounds for the four banks, against about 26 MB to pool the training features once.
* **Clipping** costs 0.002 PR-AUC (consistent but negligible) and restores FedAvg under the scaling attacker from 0.300 to 0.797 (5 of 5 seeds, t = 7.8), as effective as FedMedian (0.802).
* **Client-level DP is even less usable on this data.** At z = 1 (ε ≈ 20 over 10 rounds) the model collapses to 0.066; even ε ≈ 50 costs 0.136. The ten rounds compose more noise than the synthetic study's five, and with only a few hundred training frauds the averaged signal is easily drowned.
* **Adaptive attacks** confirm that **Krum is the weak point**: with two ALIE (z = 5) attackers it collapses on 2 of 5 seeds (0.004, 0.002) and is degraded on the other three (0.64 to 0.71). Unlike the synthetic study, FedMedian does not collapse here (0.791, −0.016, not significant), and FedAvg is again unaffected. Min-Max and weaker ALIE settings change nothing beyond about 0.02, although the small FedMedian losses under ALIE z = 1 and Min-Max are consistent across seeds (0 of 5 seeds better).

The real data therefore supports the synthetic conclusions: clipping or median aggregation for crude attacks, avoid Krum when attackers may collude adaptively, and client-level DP is impractical with four banks. As before, the banks are a simulated partition of one public dataset, not separate institutions.

---

# 8. Conclusion

In this simulated four-bank setting, federated learning reaches the accuracy of centralized training without pooling raw data, and gives a small but consistent gain over isolated training that becomes large for banks with little data or few fraud cases. It does not, in this synthetic data, show measurable sensitivity to IID-ness, FedProx or client dropout; I do not claim these effects are absent in real data, only that this study could not detect them.

Standard averaging is clearly vulnerable to a strong scaling attack, and robust aggregation removes that vulnerability for a minority of attackers at no measurable cost. None of the aggregators tested defends against half of the clients being malicious except Krum under favourable, non-adaptive conditions. The Phase E study confirmed this caveat: with two colluding adaptive (ALIE) attackers, Krum is the most fragile aggregator and median-based aggregation can also be broken, while update-norm clipping is a cheap and effective defence against crude scaling attacks. Communication cost is negligible for this model (about 38 KiB per round). Client-level differential privacy, on the other hand, is not practical with four banks: any ε below about 10 removes most of the model's usefulness. The same pattern held on the OpenML credit-card data (section 7.5), where DP was even more damaging, clipping and FedMedian again neutralised the scaling attacker, and Krum was again the aggregator that colluding ALIE attackers could break. SHAP provides model-attribution explanations of individual predictions at the local level; they are not causal explanations or proof of regulatory compliance. Absolute performance is modest (PR-AUC about 0.30, F1 about 0.04 at the 0.5 threshold) because the synthetic fraud signal is weak and the test pool is small, so these results describe relative behaviour, not deployable accuracy.

## 8.1 Future Work
Communication cost, client-level differential privacy and adaptive attackers are now covered, on synthetic and on real data (sections 7.4 and 7.5). The main open items are: secure aggregation, which hides individual updates from the server but cannot be combined directly with FedMedian or Krum and so needs a different robust-aggregation design; differential privacy at a scale where it is usable (many more clients, client sampling for privacy amplification, or record-level DP inside each bank); stronger, optimisation-based adaptive attacks; and a raw-transaction feature pipeline for the scoring API.
