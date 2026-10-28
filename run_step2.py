import os
import joblib
import matplotlib
matplotlib.use("Agg")                      # save figures without a window
import matplotlib.pyplot as plt
import pandas as pd
import shap
from core.explain import (native_importance, perm_importance, get_shap, build_lime,
                          lime_explain, summary_sentence, shap_lime_agreement)

os.makedirs("artifacts/figures", exist_ok=True)
b = joblib.load("artifacts/step1.joblib")
prep, models, Xtr, Xte, yte = b["prep"], b["models"], b["Xtr_t"], b["Xte_t"], b["yte"]
cat_cols = list(prep.transformers_[1][2])   # original categorical columns
THR = 0.5

for name in ["XGBoost", "Logistic Regression"]:
    model = models[name]
    tag = name.replace(" ", "_")
    print(f"\n===== {name} =====")

    print("Native importance (top 8):")
    print(native_importance(model, Xtr.columns).head(8).round(3))

    print("Permutation importance (top 8):")
    print(perm_importance(model, Xte, yte).head(8).round(4))

    sv = get_shap(model, Xtr, Xte.iloc[:500])
    shap.plots.beeswarm(sv, show=False)
    plt.savefig(f"artifacts/figures/beeswarm_{tag}.png", bbox_inches="tight", dpi=130)
    plt.close()

    top = pd.Series(abs(sv.values).mean(0), index=sv.feature_names).idxmax()
    shap.plots.scatter(sv[:, top], show=False)
    plt.savefig(f"artifacts/figures/dependence_{tag}.png", bbox_inches="tight", dpi=130)
    plt.close()

    # Local explanation for one test individual
    i = 0
    x_row = Xte.iloc[i]
    proba = float(model.predict_proba(Xte.iloc[[i]])[0, 1])
    sv_i = get_shap(model, Xtr, Xte.iloc[[i]])[0]
    shap.plots.waterfall(sv_i, show=False)
    plt.savefig(f"artifacts/figures/waterfall_{tag}.png", bbox_inches="tight", dpi=130)
    plt.close()

    lime_s = lime_explain(model, build_lime(Xtr), Xtr.columns, x_row)
    print("\nLIME weights:\n", lime_s.round(3))
    print("SHAP/LIME top-5 agreement:", shap_lime_agreement(sv_i, lime_s))
    print(summary_sentence(sv_i, proba, THR, cat_cols))
    print("True label:", int(yte.iloc[i]))