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

---

# 8. Conclusion

In this simulated four-bank setting, federated learning reaches the accuracy of centralized training without pooling raw data, and gives a small but consistent gain over isolated training that becomes large for banks with little data or few fraud cases. It does not, in this synthetic data, show measurable sensitivity to IID-ness, FedProx or client dropout; I do not claim these effects are absent in real data, only that this study could not detect them.

Standard averaging is clearly vulnerable to a strong scaling attack, and robust aggregation removes that vulnerability for a minority of attackers at no measurable cost. None of the aggregators tested defends against half of the clients being malicious except Krum under favourable, non-adaptive conditions. SHAP provides model-attribution explanations of individual predictions at the local level; they are not causal explanations or proof of regulatory compliance. Absolute performance is modest (PR-AUC about 0.30, F1 about 0.04 at the 0.5 threshold) because the synthetic fraud signal is weak and the test pool is small, so these results describe relative behaviour, not deployable accuracy.

## 8.1 Future Work
Future iterations of this research should investigate Differential Privacy and Secure Aggregation to defend against gradient inversion attacks, which could theoretically leak feature data. Additionally, evaluating the network bandwidth and computational latency costs associated with transmitting large model parameters will be critical for real-time production deployment.
