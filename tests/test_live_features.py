import pandas as pd
import pytest
from src.services.model import ModelService
from src.app.pages.live_stream import score_stream, running_metrics, risk_level
from src.app.pages.transaction_analyzer import score_uploaded
from src.feature_engineering import BASE_FEATURES


@pytest.fixture(scope="module")
def svc():
    s = ModelService(); s.load_model(); return s


def test_stream_scoring(svc):
    df = svc.get_test_transactions()
    sc = score_stream(svc, df, 0, 20)
    assert len(sc) == 20 and sc.probability.between(0, 1).all()
    assert running_metrics(sc)["seen"] == 20
    assert risk_level(0.9) == "HIGH" and risk_level(0.1) == "LOW"
    assert score_stream(svc, df, len(df), 5).empty


def test_upload_scoring(svc):
    df = svc.get_test_transactions()[BASE_FEATURES].head(5)
    out = score_uploaded(svc, df)
    assert len(out) == 5 and "fraud_probability" in out
    with pytest.raises(ValueError):
        score_uploaded(svc, df.drop(columns=["amount"]))
