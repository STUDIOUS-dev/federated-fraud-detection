# 1. Title Page

**Project Title:** Privacy-Aware & Explainable Federated Fraud Detection
**Student Name:** [Student Name]
**Degree/Program:** [Degree/Program]
**University:** [University]
**Academic Session:** [Academic Session]
**Project Guide:** [Guide Name]

---

# 2. Abstract

Financial fraud remains a massive challenge, but the highly sensitive nature of banking data restricts cross-institutional data sharing. Federated Learning (FL) offers a promising decentralized approach to train global models without exposing raw local datasets. In this project, I simulated a financial network of four highly heterogeneous banks (varying in transaction volume and class imbalance). I applied FedAvg and FedProx to collaboratively train a PyTorch-based Multilayer Perceptron (MLP) for fraud detection. I evaluated the architecture's resilience by introducing a malicious client aiming to poison the global model with manipulated model updates, which caused my standard FedAvg performance to collapse (PR-AUC dropped to 0.0036). To mitigate this, I deployed the FedMedian aggregation strategy, which substantially mitigated the effect of the poisoned client under the tested configuration and recovered performance (PR-AUC 0.2435). I also integrated SHAP to provide local explainability for the global model, supporting interpretability and auditability of individual model predictions. My findings demonstrate that Federated Learning can approximate centralized performance (Centralized PR-AUC: 0.2508 vs. Clean FedAvg PR-AUC: 0.2439) while keeping raw data local and evaluating robustness.

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
The system I built employs a server-client architecture using the Flower (`flwr`) framework. The underlying fraud detection model is a PyTorch-based Multilayer Perceptron (MLP). Each bank acts as a local client that trains the model on its proprietary dataset. I evaluated three primary aggregation strategies:
* **FedAvg:** Averages the local model weights, weighted by the number of samples at each client.
* **FedProx:** Introduces a proximal term to the local loss function to prevent local models from drifting too far from the global model, addressing non-IID data distributions.
* **FedMedian:** A robust aggregation strategy that takes the coordinate-wise median of the local model updates instead of the mean, reducing the influence of extreme or manipulated model updates.

## 6.3 Explainability
To address the "black-box" nature of neural networks, I integrated SHapley Additive exPlanations (SHAP). The SHAP module provides model-attribution explanations for individual transaction predictions made by the global model on local client data. It highlights the features (e.g., transaction amount, time since last transaction) driving the fraud probability, supporting interpretability and auditability of individual model predictions, but it is not a causal explanation or proof of regulatory compliance.

---

# 7. Experiments and Results

I conducted ten comprehensive experiments to evaluate the system under various conditions. I also trained a centralized model (where all data is pooled) as a reference baseline.

## 7.1 Baseline and Core Federated Performance
* **Centralized Baseline:** Achieved a global Precision-Recall Area Under Curve (PR-AUC) of **0.2508**.
* **Clean FedAvg:** Achieved a global PR-AUC of **0.2439**.
* **FedProx:** Achieved a global PR-AUC of **0.2456**.

*Finding:* My experiments indicate that federated strategies generalize better than models trained purely on isolated local data, closely matching the centralized reference baseline without requiring data pooling.

## 7.2 The Impact of Heterogeneity
I investigated the impact of realistic data distributions:
* **Unequal Client Sizes:** (Global PR-AUC: 0.4077). The results suggest that sample-count-weighted FedAvg heavily biases the global boundary toward larger banks. While the largest bank dominated the update and achieved high local performance, smaller banks suffered.
* **Class Imbalance:** (Global PR-AUC: 0.3089). Extreme class imbalances severely impact local models; however, federation allows the global model to incorporate information from other clients, addressing the difficulty of local training and evaluation for clients with zero local fraud cases.
* **Client Dropout:** (Global PR-AUC: 0.2478). Simulating unreliable network connections where clients drop out slowed convergence, emphasizing the need for over-provisioning in real-world deployments.

## 7.3 Vulnerability and Robust Defense
I simulated a scenario where a compromised bank (malicious client) intentionally sends manipulated model updates to poison the global model.
* **Poisoned FedAvg:** The standard FedAvg strategy proved vulnerable. The manipulated model updates caused a total collapse of the global model, dropping the PR-AUC to **0.0036**.
* **Poisoned FedMedian:** By switching the server aggregation strategy to FedMedian, the system substantially mitigated the effect of the poisoned client under the tested configuration, recovering the global performance to a PR-AUC of **0.2435**.

---

# 8. Conclusion

This project demonstrates the feasibility of Federated Learning for the simulated cross-institutional fraud-detection setting evaluated in this study. The FedAvg and FedProx strategies allowed my heterogeneous banks to collaboratively train a global model that performs comparably to a centralized approach, keeping raw data local. 

Furthermore, my experiments highlight critical vulnerabilities in standard FL architectures. Unequal data sizes can bias the global model against smaller institutions, and standard averaging is susceptible to data poisoning attacks. However, by employing robust aggregation strategies like FedMedian, the network can substantially mitigate the effect of the poisoned client under the tested configuration. Finally, the integration of SHAP indicates that federated models can support interpretability and auditability of individual model predictions at the local level. 

## 8.1 Future Work
Future iterations of this research should investigate Differential Privacy and Secure Aggregation to defend against gradient inversion attacks, which could theoretically leak feature data. Additionally, evaluating the network bandwidth and computational latency costs associated with transmitting large model parameters will be critical for real-time production deployment.
