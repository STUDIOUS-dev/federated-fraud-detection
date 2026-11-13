import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS_PATH = Path("results") / "real_data" / "results.json"
METRICS = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "recall_at_fpr1": "Recall @ 1% FPR"}


def load_real_data(path=RESULTS_PATH):
    """Returns the parsed real-data results dict, or None if the file is missing/unreadable."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) and "summary" in data else None


def format_summary(data: dict):
    """Builds (display table of 'mean ± std' strings, numeric mean PR-AUC series)."""
    rows, means = {}, {}
    for setup, metrics in data.get("summary", {}).items():
        row = {}
        for key, label in METRICS.items():
            ms = metrics.get(key)
            row[label] = f"{ms[0]:.3f} ± {ms[1]:.3f}" if ms else "n/a"
        rows[setup] = row
        if metrics.get("pr_auc"):
            means[setup] = float(metrics["pr_auc"][0])
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Setup"
    return table, pd.Series(means, name="PR-AUC (mean)", dtype=float)


def render(res_svc, mod_svc, exp_svc):
    st.header("Real-Data Study")
    data = load_real_data()
    if data is None:
        st.info(
            f"No real-data results found at `{RESULTS_PATH.as_posix()}`. Produce them with "
            "`python -m src.experiments.real_data_study`."
        )
        return

    frauds = data.get("test_frauds", "?")
    st.write(
        "A separate study on the public OpenML creditcard dataset: temporal split by row order, "
        f"4 simulated banks of unequal size with skewed fraud share, {data.get('seeds', '?')} seeds, "
        f"{data.get('rounds', '?')} federated rounds. Compares centralized, local-only, FedAvg and "
        "FedMedian training, with and without one malicious client."
    )

    table, means = format_summary(data)
    st.markdown("### Mean ± std over seeds")
    st.dataframe(table, use_container_width=True)
    st.markdown("### Mean PR-AUC")
    st.bar_chart(means)

    st.warning(
        f"Caveats: the test set contains only {frauds} frauds, so differences between the healthy "
        "methods are within noise. The \"banks\" are a random, skewed partition of one public "
        "dataset (row order stands in for time), not real institutions. These results are separate "
        "from the frozen synthetic results shown on other pages."
    )
