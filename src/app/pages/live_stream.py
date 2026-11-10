import time
import pandas as pd
import streamlit as st

RISK_THRESHOLD = 0.5


def risk_level(prob: float) -> str:
    if prob >= 0.8:
        return "HIGH"
    if prob >= RISK_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def score_stream(mod_svc, df: pd.DataFrame, start: int, count: int) -> pd.DataFrame:
    """Scores df[start:start+count]; returns one row per transaction."""
    chunk = df.iloc[start:start + count].copy()
    if chunk.empty:
        return pd.DataFrame(columns=["idx", "probability", "predicted", "actual", "risk"])
    probs = mod_svc.predict_proba(chunk)
    out = pd.DataFrame({
        "idx": chunk.index,
        "amount": chunk["amount"].values,
        "probability": probs,
        "predicted": (probs >= RISK_THRESHOLD).astype(int),
        "actual": chunk["is_fraud"].astype(int).values if "is_fraud" in chunk else 0,
    })
    out["risk"] = out["probability"].map(risk_level)
    return out


def running_metrics(scored: pd.DataFrame) -> dict:
    tp = int(((scored.predicted == 1) & (scored.actual == 1)).sum())
    fp = int(((scored.predicted == 1) & (scored.actual == 0)).sum())
    fn = int(((scored.predicted == 0) & (scored.actual == 1)).sum())
    return {
        "seen": len(scored),
        "alerts": int(scored.predicted.sum()),
        "fraud": int(scored.actual.sum()),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
    }


def render(res_svc, mod_svc, exp_svc):
    st.header("Live Stream")
    st.caption("Replays held-out test transactions one batch at a time, as if they were arriving live.")
    if mod_svc.model is None:
        st.error("Model is not loaded.")
        return

    df = mod_svc.get_test_transactions().sort_values("timestamp").reset_index(drop=True) \
        if "timestamp" in mod_svc.get_test_transactions() else mod_svc.get_test_transactions()

    c1, c2, c3 = st.columns(3)
    batch = c1.slider("Transactions per tick", 1, 50, 10)
    delay = c2.slider("Seconds per tick", 0.1, 2.0, 0.5)
    ticks = c3.slider("Ticks to run", 1, 50, 10)

    if "stream_pos" not in st.session_state:
        st.session_state.stream_pos = 0
        st.session_state.stream_scored = pd.DataFrame()

    b1, b2 = st.columns(2)
    run = b1.button("Run stream", type="primary")
    if b2.button("Reset"):
        st.session_state.stream_pos = 0
        st.session_state.stream_scored = pd.DataFrame()

    metrics_box, feed_box = st.empty(), st.empty()

    def draw():
        s = st.session_state.stream_scored
        if s.empty:
            return
        m = running_metrics(s)
        with metrics_box.container():
            a, b, c, d = st.columns(4)
            a.metric("Seen", m["seen"])
            b.metric("Alerts", m["alerts"])
            c.metric("Precision", "n/a" if m["precision"] != m["precision"] else f"{m['precision']:.1%}")
            d.metric("Recall", "n/a" if m["recall"] != m["recall"] else f"{m['recall']:.1%}")
        feed_box.dataframe(s[s.predicted == 1].tail(15).iloc[::-1], use_container_width=True)

    if run:
        for _ in range(ticks):
            pos = st.session_state.stream_pos
            if pos >= len(df):
                st.info("End of stream reached. Press Reset to replay.")
                break
            new = score_stream(mod_svc, df, pos, batch)
            st.session_state.stream_scored = pd.concat([st.session_state.stream_scored, new], ignore_index=True)
            st.session_state.stream_pos = pos + batch
            draw()
            time.sleep(delay)
    else:
        draw()
    st.caption(f"Position: {st.session_state.stream_pos} / {len(df)}. Alerts feed shows the latest flagged transactions.")
