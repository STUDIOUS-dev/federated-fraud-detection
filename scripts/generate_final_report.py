import json
import pandas as pd

def generate_report():
    records = []
    with open("results/summary.jsonl", "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    df = pd.DataFrame(records)
    
    # Check reproducibility
    iid_runs = df[df["exp_name"] == "exp5_iid"]
    if len(iid_runs) >= 2:
        run1 = iid_runs.iloc[-2]
        run2 = iid_runs.iloc[-1]
        print("### Reproducibility Check (exp5_iid)")
        print(f"Run 1 PR-AUC: {run1['global_metrics'].get('pr_auc')}")
        print(f"Run 2 PR-AUC: {run2['global_metrics'].get('pr_auc')}")
        print("\n")

    # Generate Markdown Table
    print("### Final Results Table\n")
    print("| Setup | Global PR-AUC | F1 | Precision | Recall |")
    print("|---|---|---|---|---|")
    
    # We want unique experiments
    unique_df = df.drop_duplicates(subset=["exp_name"], keep="last")
    
    name_mapping = {
        "exp1_centralized": "Centralized",
        "exp2_local": "Local Average",
        "exp3_fedavg": "FedAvg",
        "exp4_fedprox": "FedProx",
        "exp5_iid": "IID",
        "exp6_sizes": "Unequal Sizes",
        "exp7_imbalance": "Class Imbalance",
        "exp8_dropout": "Dropout",
        "exp9_malicious": "Poisoned FedAvg",
        "exp10_defense": "Poisoned FedMedian"
    }
    
    for _, row in unique_df.iterrows():
        name = name_mapping.get(row['exp_name'], row['exp_name'])
        metrics = row['global_metrics']
        pr_auc = metrics.get('pr_auc', 0) if metrics.get('pr_auc') is not None else 0
        f1 = metrics.get('f1', 0) if metrics.get('f1') is not None else 0
        prec = metrics.get('precision', 0) if metrics.get('precision') is not None else 0
        rec = metrics.get('recall', 0) if metrics.get('recall') is not None else 0
        print(f"| {name} | {pr_auc:.4f} | {f1:.4f} | {prec:.4f} | {rec:.4f} |")
        
    print("\n")
    print("### Validating FedMedian Result Carefully")
    exp3_df = unique_df[unique_df["exp_name"] == "exp3_fedavg"]
    exp9_df = unique_df[unique_df["exp_name"] == "exp9_malicious"]
    exp10_df = unique_df[unique_df["exp_name"] == "exp10_defense"]
    
    clean_prauc = exp3_df.iloc[0]['global_metrics'].get('pr_auc', 0) if not exp3_df.empty else 0
    poisoned_prauc = exp9_df.iloc[0]['global_metrics'].get('pr_auc', 0) if not exp9_df.empty else 0
    median_prauc = exp10_df.iloc[0]['global_metrics'].get('pr_auc', 0) if not exp10_df.empty else 0
    
    print(f"* clean FedAvg PR-AUC: {clean_prauc:.4f}")
    print(f"* poisoned FedAvg PR-AUC: {poisoned_prauc:.4f}")
    print(f"* poisoned FedMedian PR-AUC: {median_prauc:.4f}")
    print(f"* absolute degradation: {clean_prauc - poisoned_prauc:.4f}")
    print(f"* recovery amount: {median_prauc - poisoned_prauc:.4f}")
    

if __name__ == "__main__":
    generate_report()
