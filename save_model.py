import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from core.data import load_bank
from core.preprocess import build_preprocessor
from core.models import get_models, tune
from core.evaluate import metrics_at

DROP_SENSITIVE = False      # set True to serve the model without age/marital/education/job
THRESHOLD = 0.50

X, y = load_bank()
if DROP_SENSITIVE:
    X = X.drop(columns=["age", "marital", "education", "job"])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
prep = build_preprocessor(Xtr)
Xtr_t, Xte_t = prep.fit_transform(Xtr), prep.transform(Xte)

pw = (ytr == 0).sum() / (ytr == 1).sum()
model, grid = get_models(pw)["XGBoost"]
best, params = tune(model, grid, Xtr_t, ytr)
print("Best params:", params)
print({k: round(v, 3) for k, v in metrics_at(yte, best.predict_proba(Xte_t)[:, 1], THRESHOLD).items()})

num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
schema = {}
for c in X.columns:
    if c in num_cols:
        schema[c] = {"type": "number", "min": float(X[c].min()), "max": float(X[c].max()),
                     "median": float(X[c].median())}
    else:
        schema[c] = {"type": "category", "options": sorted(X[c].dropna().astype(str).unique())}

joblib.dump({
    "prep": prep, "model": best, "threshold": THRESHOLD,
    "Xtr_t": Xtr_t.sample(5000, random_state=0),   # background for LIME statistics
    "raw_columns": X.columns.tolist(), "num_cols": num_cols, "cat_cols": cat_cols,
    "schema": schema, "model_name": "XGBoost",
}, "artifacts/best_model.joblib", compress=3)
print("Saved artifacts/best_model.joblib")