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


def test_non_finite_rejected(rows):
    # raw JSON so NaN/Infinity literals are actually sent
    for lit in ("NaN", "Infinity", "-Infinity"):
        body = '{"transactions": [{' + ", ".join(
            f'"{k}": {lit if k == "amount" else v}' for k, v in rows[0].items()) + "}]}"
        r = client.post("/score", content=body, headers={"content-type": "application/json"})
        assert r.status_code == 422, lit


def test_explain_cap(rows):
    from src.api import MAX_EXPLAIN_TRANSACTIONS
    many = [rows[0]] * (MAX_EXPLAIN_TRANSACTIONS + 1)
    r = client.post("/score", json={"transactions": many, "explain": True})
    assert r.status_code == 422 and str(MAX_EXPLAIN_TRANSACTIONS) in r.json()["detail"]
    # without explain the same batch is fine
    assert client.post("/score", json={"transactions": many}).status_code == 200


def test_score_503_when_model_unavailable(rows, monkeypatch):
    import src.api as api

    def boom():
        raise FileNotFoundError("model checkpoint missing")

    monkeypatch.setattr(api, "_services", boom)
    r = client.post("/score", json={"transactions": rows[:1]})
    assert r.status_code == 503 and "model unavailable" in r.json()["detail"]
