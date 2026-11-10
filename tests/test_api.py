import pytest
from fastapi.testclient import TestClient
from src.api import app
from src.feature_engineering import BASE_FEATURES
from src.services.model import ModelService

client = TestClient(app)


@pytest.fixture(scope="module")
def rows():
    ms = ModelService(); ms.load_model()
    return ms.get_test_transactions()[BASE_FEATURES].head(3).to_dict("records")


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["features"] == BASE_FEATURES


def test_score_batch(rows):
    r = client.post("/score", json={"transactions": rows})
    assert r.status_code == 200
    res = r.json()["results"]
    assert len(res) == 3 and all(0 <= x["fraud_probability"] <= 1 for x in res)


def test_score_explain_and_threshold(rows):
    r = client.post("/score", json={"transactions": rows[:1], "explain": True, "threshold": 0.0})
    x = r.json()["results"][0]
    assert x["predicted_fraud"] is True and x["explanation"]


def test_validation_errors(rows):
    bad = dict(rows[0]); bad.pop("amount")
    assert client.post("/score", json={"transactions": [bad]}).status_code == 422
    assert client.post("/score", json={"transactions": []}).status_code == 422
