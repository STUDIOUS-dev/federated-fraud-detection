"""Shared display labels for experiment records."""

NON_FEDERATED = {
    "exp1_centralized": "Centralized (no federation)",
    "exp2_local": "Local only (no federation)",
}


def is_federated(exp_name) -> bool:
    return bool(exp_name) and exp_name not in NON_FEDERATED


def algorithm_label(record: dict) -> str:
    """Human-readable aggregation label; baselines never claim an FL algorithm."""
    name = record.get("exp_name")
    if name in NON_FEDERATED:
        return NON_FEDERATED[name]
    return record.get("aggregation") or "N/A"
