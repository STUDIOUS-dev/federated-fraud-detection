import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.app.labels import algorithm_label, is_federated
from src.app.pages.real_data import format_summary, load_real_data
from src.app.pages.transaction_analyzer import score_uploaded
from src.feature_engineering import BASE_FEATURES
from src.services.model import ModelService

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "src" / "dashboard.py"


def _page_names():
    src = DASHBOARD.read_text(encoding="utf-8")
    block = src.split("PAGES = {", 1)[1].split("}", 1)[0]
    return re.findall(r'^\s*"([^"]+)":', block, flags=re.M)


def test_every_page_renders():
    at = AppTest.from_file(str(DASHBOARD))
    at.run(timeout=180)
    assert not at.exception, f"startup: {at.exception}"
    options = list(at.sidebar.radio[0].options)
    assert options == _page_names()
    assert "Real-Data Study" in options and "Live Stream" in options
    for name in options:
        at.sidebar.radio[0].set_value(name).run(timeout=180)
        assert not at.exception, f"{name}: {at.exception}"


def test_home_has_no_raw_mermaid_or_callout():
    at = AppTest.from_file(str(DASHBOARD))
    at.run(timeout=180)
    text = " ".join(m.value for m in at.markdown)
    assert "mermaid" not in text and "[!NOTE]" not in text


def test_real_data_loader(tmp_path):
    assert load_real_data(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert load_real_data(bad) is None
    good = tmp_path / "r.json"
    good.write_text(json.dumps({
        "seeds": 2, "rounds": 3, "test_frauds": 10,
        "summary": {"FedAvg": {"pr_auc": [0.8, 0.01], "roc_auc": [0.9, 0.02],
                               "recall_at_fpr1": [0.7, 0.03]}},
    }))
    data = load_real_data(good)
    table, means = format_summary(data)
    assert table.loc["FedAvg", "PR-AUC"] == "0.800 ± 0.010"
    assert list(table.columns) == ["PR-AUC", "ROC-AUC", "Recall @ 1% FPR"]
    assert means["FedAvg"] == 0.8


def test_real_data_loader_on_frozen_results():
    data = load_real_data(ROOT / "results" / "real_data" / "results.json")
    if data is None:
        pytest.skip("real-data results not present")
    table, means = format_summary(data)
    assert len(table) == len(means) == len(data["summary"])


def test_labels():
    assert not is_federated("exp1_centralized") and not is_federated("exp2_local")
    assert not is_federated(None)
    assert is_federated("exp3_fedavg")
    for name in ("exp1_centralized", "exp2_local"):
        assert "FedAvg" not in algorithm_label({"exp_name": name, "aggregation": "FedAvg"})
        assert "no federation" in algorithm_label({"exp_name": name, "aggregation": "FedAvg"})
    assert algorithm_label({"exp_name": "exp10_defense", "aggregation": "FedMedian"}) == "FedMedian"
    assert algorithm_label({"exp_name": "x"}) == "N/A"


@pytest.fixture(scope="module")
def svc():
    s = ModelService()
    s.load_model()
    return s


def test_score_uploaded_validation(svc):
    base = svc.get_test_transactions()[BASE_FEATURES].head(4).reset_index(drop=True)

    with pytest.raises(ValueError, match="Missing required columns"):
        score_uploaded(svc, base.drop(columns=["amount"]))
    with pytest.raises(ValueError, match="no rows"):
        score_uploaded(svc, base.iloc[0:0])

    bad = base.astype(object)
    bad.loc[1, "amount"] = "abc"
    with pytest.raises(ValueError, match="amount"):
        score_uploaded(svc, bad)

    nan = base.copy()
    nan.loc[2, BASE_FEATURES[1]] = np.nan
    with pytest.raises(ValueError, match=BASE_FEATURES[1]):
        score_uploaded(svc, nan)

    inf = base.copy()
    inf.loc[0, "amount"] = np.inf
    with pytest.raises(ValueError, match="amount"):
        score_uploaded(svc, inf)

    ok = base.copy()
    ok["note"] = ["a", "b", "c", "d"]
    ok["amount"] = ok["amount"].astype(str)  # numeric strings are coerced
    out = score_uploaded(svc, ok)
    assert "note" in out.columns and list(out["note"]) == ["a", "b", "c", "d"]
    assert out["fraud_probability"].between(0, 1).all()
