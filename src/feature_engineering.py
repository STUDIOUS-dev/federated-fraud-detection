from __future__ import annotations
import pandas as pd

BASE_FEATURES = [
    "amount", "hour", "day_of_week", "is_weekend",
    "customer_mean_before", "customer_count_before",
    "customer_amount_std_before", "amount_deviation", "minutes_since_customer_tx",
    "terminal_count_before", "terminal_mean_amount_before", "is_new_terminal",
    "customer_tx_1h", "customer_amount_1h",
]


def add_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.sort_values(["customer_id", "timestamp"]).copy()
    x["hour"] = x["timestamp"].dt.hour.astype("int16")
    x["day_of_week"] = x["timestamp"].dt.dayofweek.astype("int16")
    x["is_weekend"] = (x["day_of_week"] >= 5).astype("int8")

    # Shift first, then expanding/rolling: current row is never used to build its own features.
    g = x.groupby("customer_id", sort=False)
    x["customer_mean_before"] = g["amount"].transform(lambda s: s.shift(1).expanding().mean())
    x["customer_count_before"] = g.cumcount()
    x["customer_amount_std_before"] = g["amount"].transform(lambda s: s.shift(1).expanding().std())
    x["amount_deviation"] = (x["amount"] - x["customer_mean_before"]).abs()
    x["minutes_since_customer_tx"] = g["timestamp"].diff().dt.total_seconds().div(60)

    tg = x.groupby("terminal_id", sort=False)
    x["terminal_count_before"] = tg.cumcount()
    x["terminal_mean_amount_before"] = tg["amount"].transform(lambda s: s.shift(1).expanding().mean())
    x["is_new_terminal"] = (x["terminal_count_before"] == 0).astype("int8")

    # Time-window features are shifted so the current transaction is excluded.
    x = x.set_index("timestamp")
    x["customer_tx_1h"] = (
        x.groupby("customer_id")["amount"]
         .rolling("1h", closed="left").count()
         .reset_index(level=0, drop=True)
    ).to_numpy()
    x["customer_amount_1h"] = (
        x.groupby("customer_id")["amount"]
         .rolling("1h", closed="left").sum()
         .reset_index(level=0, drop=True)
    ).to_numpy()
    x = x.reset_index()

    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    x[BASE_FEATURES] = x[BASE_FEATURES].fillna(0.0)
    return x.sort_values("timestamp").reset_index(drop=True)
