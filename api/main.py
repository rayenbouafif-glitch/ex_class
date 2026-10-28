import pathlib
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.explain import (get_shap, build_lime, lime_explain, describe, summary_sentence)

ROOT = pathlib.Path(__file__).resolve().parent.parent
B = joblib.load(ROOT / "artifacts" / "best_model.joblib")
PREP, MODEL, THR = B["prep"], B["model"], B["threshold"]
COLS, NUM, CAT = B["raw_columns"], B["num_cols"], B["cat_cols"]
XTR = B["Xtr_t"]
LIME = build_lime(XTR)

app = FastAPI(
    title="Explainable Classification API",
    description="Predicts term-deposit subscription (Bank Marketing) and explains each "
                "prediction with SHAP and LIME. Send raw client features; preprocessing "
                "is applied server-side.",
    version="1.0",
)


class Individual(BaseModel):
    features: dict = Field(
        ...,
        examples=[{"age": 40, "job": "technician", "marital": "married", "education": "secondary",
                   "default": "no", "balance": 640, "housing": "no", "loan": "no",
                   "contact": "cellular", "day_of_week": 12, "month": "mar", "campaign": 1,
                   "pdays": -1, "previous": 0, "poutcome": "success"}],
    )
    threshold: float | None = Field(None, ge=0, le=1, description="Decision threshold (default: model's)")
    top_k: int = Field(5, ge=1, le=15, description="Number of reason codes")


def to_frame(ind: Individual) -> pd.DataFrame:
    unknown = set(ind.features) - set(COLS)
    if unknown:
        raise HTTPException(422, f"Unknown features: {sorted(unknown)}. Expected: {COLS}")
    df = pd.DataFrame([ind.features]).reindex(columns=COLS)
    df = df.astype(object).where(df.notna(), np.nan)
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def transform(ind: Individual) -> pd.DataFrame:
    return PREP.transform(to_frame(ind))


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "model": B["model_name"], "n_features": XTR.shape[1], "threshold": THR}


@app.get("/schema", tags=["meta"])
def schema():
    """Expected raw inputs (names, types, ranges, categories)."""
    return B["schema"]


@app.post("/predict", tags=["prediction"])
def predict(ind: Individual):
    p = float(MODEL.predict_proba(transform(ind))[0, 1])
    thr = ind.threshold if ind.threshold is not None else THR
    return {"probability": p, "threshold": thr, "decision": "POSITIVE" if p >= thr else "NEGATIVE"}


@app.post("/explain/shap", tags=["explanation"])
def explain_shap(ind: Individual):
    x = transform(ind)
    p = float(MODEL.predict_proba(x)[0, 1])
    thr = ind.threshold if ind.threshold is not None else THR
    sv = get_shap(MODEL, XTR, x)[0]
    order = np.argsort(-np.abs(sv.values))[: ind.top_k]
    reasons = [{"feature": sv.feature_names[i],
                "label": describe(sv.feature_names[i], float(sv.data[i]), CAT),
                "shap": float(sv.values[i]),
                "direction": "up" if sv.values[i] > 0 else "down"} for i in order]
    return {
        "probability": p,
        "decision": "POSITIVE" if p >= thr else "NEGATIVE",
        "base_value": float(sv.base_values),
        "unit": "log-odds",
        "summary": summary_sentence(sv, p, thr, CAT),
        "reason_codes": reasons,
        "contributions": {n: float(v) for n, v in zip(sv.feature_names, sv.values)},
    }


@app.post("/explain/lime", tags=["explanation"])
def explain_lime(ind: Individual):
    x = transform(ind)
    s = lime_explain(MODEL, LIME, XTR.columns, x.iloc[0], n=max(ind.top_k, 10))
    return {"lime": {k: float(v) for k, v in s.items()}}