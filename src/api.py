"""Scoring API: POST /score returns fraud probability (+ optional SHAP reasons).

Run: uvicorn src.api:app --port 8000
Inputs are the 14 engineered features (see src.feature_engineering.BASE_FEATURES).
Feature values must be finite (NaN/Infinity are rejected with 422). At most 1000
transactions per request; with explain=true at most MAX_EXPLAIN_TRANSACTIONS (50)
are allowed, since SHAP runs once per row (422 above that). /score returns 503 if
the model artifacts cannot be loaded.
"""
from functools import lru_cache
from typing import Annotated, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, create_model

from src.feature_engineering import BASE_FEATURES
from src.services.model import ModelService
from src.services.explainability import ExplainabilityService

MAX_EXPLAIN_TRANSACTIONS = 50

Transaction = create_model(
    "Transaction", **{f: (Annotated[float, Field(allow_inf_nan=False)], ...) for f in BASE_FEATURES}
)


class ScoreRequest(BaseModel):
    transactions: List[Transaction] = Field(..., min_length=1, max_length=1000)
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    explain: bool = False


class ScoreResult(BaseModel):
    fraud_probability: float
    predicted_fraud: bool
    risk: str
    explanation: Optional[str] = None


class ScoreResponse(BaseModel):
    results: List[ScoreResult]


app = FastAPI(title="Federated Fraud Detection API", version="1.0")


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    # Drop the echoed "input"/"ctx": a NaN/inf input cannot be JSON-serialised
    # and would turn the 422 into a 500.
    detail = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")}
              for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


@lru_cache(maxsize=1)
def _services():
    ms = ModelService()
    ms.load_model()
    return ms, ExplainabilityService(ms)


def _risk(p: float, threshold: float) -> str:
    return "HIGH" if p >= max(0.8, threshold) else "MEDIUM" if p >= threshold else "LOW"


@app.get("/health")
def health():
    try:
        _services()
        return {"status": "ok", "features": BASE_FEATURES}
    except Exception as e:
        raise HTTPException(503, f"model unavailable: {e}")


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    try:
        ms, es = _services()
    except Exception as e:
        raise HTTPException(503, f"model unavailable: {e}")
    if req.explain and len(req.transactions) > MAX_EXPLAIN_TRANSACTIONS:
        raise HTTPException(
            422, f"explain=true supports at most {MAX_EXPLAIN_TRANSACTIONS} transactions per request "
                 f"(got {len(req.transactions)})")
    df = pd.DataFrame([t.model_dump() for t in req.transactions])[BASE_FEATURES]
    probs = ms.predict_proba(df)
    results = []
    for i, p in enumerate(probs):
        p = float(p)
        text = None
        if req.explain:
            text = es.explain(df.iloc[[i]].copy())["explanation"]
        results.append(ScoreResult(fraud_probability=p, predicted_fraud=p >= req.threshold,
                                   risk=_risk(p, req.threshold), explanation=text))
    return ScoreResponse(results=results)
