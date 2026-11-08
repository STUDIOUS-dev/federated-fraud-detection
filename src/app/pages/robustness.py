import json
from pathlib import Path
import streamlit as st
import pandas as pd

STUDY_AGG = Path("results/study/aggregate.json")


def _study_table():
    """Mean +- std over seeds from the multi-seed study (if it has been run)."""
    if not STUDY_AGG.exists():
        return None
    sc = json.loads(STUDY_AGG.read_text())["scenarios"]
    rows = []
    for attack in ("scale", "signflip", "labelflip"):
        row = {"Attack (1 of 4 banks)": attack}
        for label, agg in (("FedAvg", "fedavg"), ("FedMedian", "fedmedian"),
                           ("FedTrimmedAvg", "fedtrimmedavg"), ("Krum", "krum")):
            g = sc.get(f"attack_{attack}_x1_{agg}", {}).get("global", {}).get("pr_auc")
            row[label] = f"{g['mean']:.3f} ± {g['std']:.3f}" if g and g["mean"] is not None else "n/a"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Attack (1 of 4 banks)")


def render(res_svc, mod_svc, exp_svc):
    st.header("Robustness to Malicious Poisoning")

    st.markdown("""
    This page demonstrates the vulnerability of standard federated learning to malicious model updates,
    and how robust aggregation mechanisms mitigate the attack.

    > **Experiment Context:** A single malicious client (Bank D) poisons its model updates before sending them
    > to the coordinator. The headline run below uses the *weight-scaling* attack (weights multiplied by -10).
    """)

    st.divider()

    try:
        clean = res_svc.get_experiment_metrics("exp3_fedavg").get("pr_auc")
        poisoned = res_svc.get_experiment_metrics("exp9_malicious").get("pr_auc")
        defense = res_svc.get_experiment_metrics("exp10_defense").get("pr_auc")
        if any(v is None for v in [clean, poisoned, defense]):
            st.error("Robustness experiments (exp3, exp9, exp10) are missing from results/summary.jsonl. "
                     "Re-run them with `python scripts/run_all_experiments.py`.")
            return

        st.markdown("### Single-seed demonstration (seed 42)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Clean FedAvg", f"{clean:.4f}")
        c2.metric("Poisoned FedAvg", f"{poisoned:.4f}", delta=f"{poisoned - clean:.4f}", delta_color="inverse")
        c3.metric("Poisoned FedMedian", f"{defense:.4f}", delta=f"{defense - poisoned:.4f}")

        st.bar_chart(pd.Series({
            "Clean FedAvg": clean,
            "Poisoned FedAvg": poisoned,
            "Poisoned FedMedian": defense
        }))

        table = _study_table()
        if table is not None:
            st.markdown("### Multi-seed study: PR-AUC (mean ± std over seeds)")
            st.dataframe(table, use_container_width=True)
            st.caption("From `results/study/aggregate.json`. See docs/STUDY_RESULTS.md for 2-attacker results and "
                       "paired comparisons.")

        st.info("Under the tested attacks, standard FedAvg degrades sharply; robust aggregators help to a degree that "
                "depends on the attack and the number of attackers. See the multi-seed table above and "
                "docs/STUDY_RESULTS.md before drawing conclusions from the single-seed bars.")

    except Exception as e:
        st.error(f"Could not load robustness metrics: {e}")
