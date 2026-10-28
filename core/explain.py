import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lime.lime_tabular import LimeTabularExplainer

TREE_TYPES = (DecisionTreeClassifier, RandomForestClassifier, XGBClassifier)


# ---------- Global ----------
def native_importance(model, names):
    """Coefficients (logistic) or impurity importance (trees)."""
    if hasattr(model, "coef_"):
        vals = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    else:
        return None  # kNN has no native importance
    return pd.Series(vals, index=names).sort_values(key=np.abs, ascending=False)


def perm_importance(model, Xte, yte, n_max=3000, n_repeats=5):
    idx = np.random.RandomState(0).choice(len(Xte), min(n_max, len(Xte)), replace=False)
    r = permutation_importance(model, Xte.iloc[idx], yte.iloc[idx],
                               scoring="roc_auc", n_repeats=n_repeats,
                               random_state=0, n_jobs=-1)
    return pd.Series(r.importances_mean, index=Xte.columns).sort_values(ascending=False)


def get_shap(model, Xtr, X, n_background=50):
    """Returns a shap.Explanation (n_samples x n_features) for the positive class."""
    if isinstance(model, LogisticRegression):
        masker = shap.maskers.Independent(Xtr, max_samples=200)
        sv = shap.LinearExplainer(model, masker)(X)
    elif isinstance(model, TREE_TYPES):
        sv = shap.TreeExplainer(model)(X, check_additivity=False)
    else:  # kNN and others: slow, so use a small background and few rows
        f = lambda d: model.predict_proba(pd.DataFrame(d, columns=Xtr.columns))[:, 1]
        sv = shap.Explainer(f, shap.sample(Xtr, n_background, random_state=0))(X)
    if sv.values.ndim == 3:        # RF / decision tree: keep positive class
        sv = sv[:, :, 1]
    return sv


# ---------- Local ----------
def build_lime(Xtr):
    return LimeTabularExplainer(Xtr.values, feature_names=list(Xtr.columns),
                                class_names=["No", "Yes"], mode="classification",
                                discretize_continuous=False, random_state=0)


def lime_explain(model, lime_exp, Xtr_cols, x_row, n=10):
    """Returns a Series feature -> LIME weight for the positive class."""
    f = lambda a: model.predict_proba(pd.DataFrame(a, columns=Xtr_cols))
    e = lime_exp.explain_instance(x_row.values, f, num_features=n, labels=(1,))
    return pd.Series({Xtr_cols[i]: w for i, w in e.as_map()[1]})


def describe(name, value, cat_cols):
    """Readable label that takes the feature's actual value into account."""
    for c in cat_cols:
        if name.startswith(c + "_"):
            cat = name[len(c) + 1:]
            return f"{c} = {cat}" if value > 0.5 else f"{c} ≠ {cat}"
    # numeric features are standardized: sign tells above/below average
    return f"{name} {'above' if value > 0 else 'below'} average"


def summary_sentence(sv_row, proba, thr, cat_cols, target_label="subscribing"):
    order = np.argsort(-np.abs(sv_row.values))[:3]
    lab = lambda i: describe(sv_row.feature_names[i], sv_row.data[i], cat_cols)
    up = [lab(i) for i in order if sv_row.values[i] > 0]
    down = [lab(i) for i in order if sv_row.values[i] < 0]
    level = "HIGH" if proba >= thr else "LOW"
    s = f"The likelihood of {target_label} is {level} ({proba:.0%}, threshold {thr:.0%})"
    if up:
        s += f"; pushed up by: {', '.join(up)}"
    if down:
        s += f"; pushed down by: {', '.join(down)}"
    return s + "."

def shap_lime_agreement(sv_row, lime_series, k=5):
    """Overlap of the top-k features between SHAP and LIME (0 to 1)."""
    s = set(pd.Series(sv_row.values, index=sv_row.feature_names).abs().nlargest(k).index)
    l = set(lime_series.abs().nlargest(k).index)
    return len(s & l) / k
def shap_by_variable(sv, cat_cols):
    """Mean |SHAP| aggregated from one-hot columns back to original variables, as % share."""
    imp = pd.Series(np.abs(sv.values).mean(0), index=sv.feature_names)
    def orig(n):
        for c in cat_cols:
            if n.startswith(c + "_"):
                return c
        return n
    g = imp.groupby(orig).sum().sort_values(ascending=False)
    return (100 * g / g.sum()).round(1)