from sklearn.model_selection import train_test_split
from core.data import load_bank
from core.preprocess import build_preprocessor
from core.models import get_models, tune
from core.evaluate import metrics_at

X, y = load_bank()
SENS = ["age", "marital", "education", "job"]
variants = {"All variables": list(X.columns),
            "Without sensitive": [c for c in X.columns if c not in SENS]}

for label, cols in variants.items():
    Xtr, Xte, ytr, yte = train_test_split(X[cols], y, test_size=0.2, stratify=y, random_state=42)
    prep = build_preprocessor(Xtr)
    Xtr_t, Xte_t = prep.fit_transform(Xtr), prep.transform(Xte)
    pw = (ytr == 0).sum() / (ytr == 1).sum()
    for name in ["Logistic Regression", "XGBoost"]:
        model, grid = get_models(pw)[name]
        best, _ = tune(model, grid, Xtr_t, ytr)
        m = metrics_at(yte, best.predict_proba(Xte_t)[:, 1])
        print(f"{label:20s} {name:20s} AUC={m['AUC']:.3f}  PR-AUC={m['PR-AUC']:.3f}")