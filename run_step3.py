import joblib
from core.explain import get_shap, shap_by_variable

b = joblib.load("artifacts/step1.joblib")
prep, models, Xtr, Xte = b["prep"], b["models"], b["Xtr_t"], b["Xte_t"]
cat_cols = list(prep.transformers_[1][2])
SENSITIVE = ["age", "marital", "job", "education"]

for name in ["XGBoost", "Logistic Regression"]:
    sv = get_shap(models[name], Xtr, Xte.iloc[:1000])
    share = shap_by_variable(sv, cat_cols)
    print(f"\n===== {name}: share of total SHAP importance (%) =====")
    print(share)
    print("Sensitive variables total:", round(share[share.index.isin(SENSITIVE)].sum(), 1), "%")