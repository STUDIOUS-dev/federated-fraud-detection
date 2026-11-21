"""Phase E page: communication cost, update clipping / differential privacy, adaptive attacks.
Reads results/study_e/aggregate.json (written by `python -m src.experiments.study_e`)."""
import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st

AGGREGATE_PATH = Path("results") / "study_e" / "aggregate.json"
AGGS = [("FedAvg", "fedavg"), ("FedMedian", "fedmedian"), ("FedTrimmedAvg", "fedtrimmedavg"), ("Krum", "krum")]
ROUNDS = [(1, "fedavg_rounds1"), (2, "fedavg_rounds2"), (3, "fedavg_rounds3"), (5, "fedavg"), (10, "fedavg_rounds10")]
DP = ["fedavg_clip0.5", "dp_z0.5", "dp_z1.0", "dp_z2.0", "dp_z4.0"]


def load_phase_e(path=AGGREGATE_PATH):
    """Parsed aggregate.json scenarios dict, or None if missing/unreadable."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    sc = data.get("scenarios") if isinstance(data, dict) else None
    return sc if isinstance(sc, dict) and sc else None


def _ms(entry, metric="pr_auc"):
    s = (entry or {}).get("global", {}).get(metric) or {}
    return s.get("mean"), s.get("std")


def _fmt(entry, metric="pr_auc"):
    m, s = _ms(entry, metric)
    return "n/a" if m is None else f"{m:.3f} ± {s:.3f}"


def communication_table(sc):
    rows = []
    for r, name in ROUNDS:
        e = sc.get(name)
        if e:
            b = e.get("communication_bytes")
            rows.append({"Rounds": r, "Traffic (KiB)": None if b is None else round(b / 1024, 1),
                         "PR-AUC": _ms(e)[0]})
    return pd.DataFrame(rows)


def privacy_table(sc):
    rows = []
    for name in DP:
        e = sc.get(name)
        if e:
            eps = e.get("epsilon", math.inf)
            rows.append({"Setting": name, "ε (δ=1e-5)": "∞ (no noise)" if eps is None or math.isinf(eps)
                         else f"{eps:.1f}", "PR-AUC": _fmt(e), "Recall @ 1% FPR": _fmt(e, "recall_at_fpr1"),
                         "_mean": _ms(e)[0]})
    return pd.DataFrame(rows)


def attack_table(sc, k):
    rows = {}
    for attack in ("alie", "alie_z3", "minmax"):
        row = {label: _fmt(sc.get(f"adaptive_{attack}_{k}_{a}")) for label, a in AGGS}
        if any(v != "n/a" for v in row.values()):
            rows[attack] = row
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "Attack"
    return df


def render(res_svc, mod_svc, exp_svc):
    st.header("Privacy, Cost & Adaptive Attacks (Phase E)")
    sc = load_phase_e()
    if sc is None:
        st.info(f"No Phase E results found at `{AGGREGATE_PATH.as_posix()}`. "
                "Produce them with `python -m src.experiments.study_e`.")
        return
    base = sc.get("fedavg")
    st.write("Multi-seed results (mean ± std over seeds) on the same reference test pool as the main study. "
             f"Clean FedAvg reference: PR-AUC **{_fmt(base)}**. Full tables and paired tests: "
             "`docs/PHASE_E_RESULTS.md`.")

    st.subheader("Communication cost")
    comm = communication_table(sc)
    if not comm.empty:
        st.dataframe(comm, hide_index=True, use_container_width=True)
        st.caption("Bytes on the wire for all four banks, uploads plus downloads. Each update is the "
                   "1,025-parameter model, independent of how many transactions a bank holds.")

    st.subheader("Update clipping and client-level differential privacy")
    priv = privacy_table(sc)
    if not priv.empty:
        st.dataframe(priv.drop(columns="_mean"), hide_index=True, use_container_width=True)
        st.bar_chart(priv.set_index("Setting")["_mean"].rename("PR-AUC (mean)"))
        st.caption("Updates clipped to L2 norm 0.5; Gaussian noise std = z·0.5/m added to the average. "
                   "ε protects each bank's entire contribution (zCDP accounting, 5 rounds, no amplification).")

    st.subheader("Adaptive (omniscient) attacks vs aggregators — PR-AUC")
    for k, label in (("x1", "1 attacker of 4"), ("x2", "2 colluding attackers of 4")):
        t = attack_table(sc, k)
        if not t.empty:
            st.markdown(f"**{label}**")
            st.dataframe(t, use_container_width=True)

    st.warning("Caveats: synthetic data with ~50 fraud cases per test pool; only 4 clients, which makes "
               "client-level DP very costly and robust aggregation fragile; attackers here know every honest "
               "update, which is the worst case rather than a typical one.")
