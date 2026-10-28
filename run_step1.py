import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from core.data import load_bank
from core.preprocess import build_preprocessor
from core.models import get_models, tune
from core.evaluate import metrics_at

X, y = load_bank()
print("Shape:", X.shape, "| positive rate:", round(y.mean(), 3))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

prep = build_preprocessor(Xtr)
Xtr_t = prep.fit_transform(Xtr)   # fit on train only
Xte_t = prep.transform(Xte)
print("Features after encoding:", Xtr_t.shape[1])

pos_weight = (ytr == 0).sum() / (ytr == 1).sum()
rows, fitted = [], {}
for name, (model, grid) in get_models(pos_weight).items():
    print("Training", name, "...")
    best, params = tune(model, grid, Xtr_t, ytr)
    proba = best.predict_proba(Xte_t)[:, 1]
    rows.append({"Model": name, **metrics_at(yte, proba), "Best params": params})
    fitted[name] = best

res = pd.DataFrame(rows).set_index("Model")
print(res.drop(columns="Best params").round(3).sort_values("AUC", ascending=False))

joblib.dump({"prep": prep, "models": fitted, "Xtr_t": Xtr_t, "Xte_t": Xte_t, "yte": yte},
            "artifacts/step1.joblib")