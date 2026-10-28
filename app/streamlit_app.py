import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import shap
from core.explain import (native_importance, perm_importance, get_shap, build_lime,
                          lime_explain, summary_sentence, shap_lime_agreement, shap_by_variable)
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

from core.data import load_bank
from core.preprocess import build_preprocessor
from core.models import get_models, tune
from core.evaluate import metrics_at

st.set_page_config(page_title="Explainable Classification", layout="wide")
st.title("Explainable Classification: Bank Marketing")
S = st.session_state

@st.cache_data(show_spinner="Downloading dataset...")
def cached_bank():
    return load_bank()

tabs = st.tabs(["1 Data", "2 Preparation", "3 Modeling", "4 Evaluation", "5 XAI"])

# ---------------- 1. DATA ----------------
with tabs[0]:
    src = st.radio("Source", ["Bank Marketing (auto-download)", "Upload CSV"], horizontal=True)
    if src.startswith("Bank"):
        X, y = cached_bank()
        S.X, S.y = X, y
    else:
        f = st.file_uploader("CSV file", type="csv")
        if f is not None:
            raw = pd.read_csv(f)
            target = st.selectbox("Target column", raw.columns)
            pos = st.selectbox("Positive class", raw[target].dropna().unique())
            S.y = (raw[target] == pos).astype(int)
            S.X = raw.drop(columns=[target])

    if "X" in S:
        X, y = S.X, S.y
        df = X.assign(target=y)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Features", X.shape[1])
        c3.metric("Positive rate", f"{y.mean():.1%}")

        st.subheader("Preview")
        st.dataframe(df.head(50), use_container_width=True)
        a, b = st.columns(2)
        a.subheader("Descriptive statistics")
        a.dataframe(df.describe().T, use_container_width=True)
        b.subheader("Missing values (%)")
        b.dataframe((X.isna().mean() * 100).round(2).rename("missing %"), use_container_width=True)

        st.subheader("Class balance")
        st.plotly_chart(px.bar(y.value_counts().rename_axis("class").reset_index(name="count"),
                               x="class", y="count"), use_container_width=True)

        st.subheader("Distribution of a variable")
        col = st.selectbox("Variable", X.columns, key="dist")
        if pd.api.types.is_numeric_dtype(X[col]):
            st.plotly_chart(px.histogram(df, x=col, color="target", barmode="overlay", opacity=0.6),
                            use_container_width=True)
        else:
            st.plotly_chart(px.histogram(df, x=col, color="target", barmode="group"),
                            use_container_width=True)

        st.subheader("Positive-class rate per variable")
        col2 = st.selectbox("Variable", X.columns, key="rate")
        s = X[col2]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 12:
            s = pd.qcut(s, 10, duplicates="drop").astype(str)
        rate = y.groupby(s.astype(str)).mean().reset_index()
        rate.columns = [col2, "positive rate"]
        st.plotly_chart(px.bar(rate, x=col2, y="positive rate"), use_container_width=True)

        st.subheader("Correlations (numeric variables)")
        corr = df.select_dtypes("number").corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1, aspect="auto"), use_container_width=True)

# ---------------- 2. PREPARATION ----------------
with tabs[1]:
    if "X" not in S:
        st.info("Load data in tab 1 first.")
    else:
        feats = st.multiselect("Explanatory variables", list(S.X.columns), default=list(S.X.columns))
        c1, c2, c3 = st.columns(3)
        imp = c1.selectbox("Numeric missing values", ["median", "mean", "most_frequent"])
        scale = c2.checkbox("Standardize numeric variables", value=True)
        test_size = c3.slider("Test size", 0.1, 0.5, 0.2, 0.05)
        st.caption("Categorical variables: missing values become 'missing', then one-hot encoding.")
        if st.button("Prepare data", type="primary"):
            if not feats:
                st.error("Select at least one variable.")
            else:
                Xtr, Xte, ytr, yte = train_test_split(S.X[feats], S.y, test_size=test_size,
                                                      stratify=S.y, random_state=42)
                prep = build_preprocessor(Xtr, num_impute=imp, scale=scale)
                S.prep = prep
                S.Xtr_t, S.Xte_t = prep.fit_transform(Xtr), prep.transform(Xte)
                S.ytr, S.yte = ytr, yte
                S.Xtr_raw, S.Xte_raw = Xtr, Xte   # raw rows, needed for the individual form
                S.pop("xai_cache", None)
                S.pop("models", None)
        if "Xtr_t" in S:
            st.success(f"Train: {S.Xtr_t.shape} | Test: {S.Xte_t.shape}")
            st.dataframe(S.Xtr_t.head(10), use_container_width=True)

# ---------------- 3. MODELING ----------------
with tabs[2]:
    if "Xtr_t" not in S:
        st.info("Prepare the data in tab 2 first.")
    else:
        pw = (S.ytr == 0).sum() / (S.ytr == 1).sum()
        catalog = get_models(pw)
        chosen = st.multiselect("Models to train", list(catalog), default=["Logistic Regression", "XGBoost"])
        n_iter = st.slider("Random-search iterations", 2, 15, 6)
        grids = {}
        for name in chosen:
            with st.expander(f"Hyperparameters: {name}"):
                g = {}
                for p, vals in catalog[name][1].items():
                    sel = st.multiselect(p, vals, default=vals, format_func=str, key=f"{name}_{p}")
                    g[p] = sel or vals
                grids[name] = g
        if st.button("Train models", type="primary") and chosen:
            S.models, S.probas, S.params = {}, {}, {}
            S.pop("xai_cache", None)
            bar = st.progress(0.0)
            for i, name in enumerate(chosen):
                with st.spinner(f"Training {name}..."):
                    best, params = tune(catalog[name][0], grids[name], S.Xtr_t, S.ytr, n_iter=n_iter)
                S.models[name] = best
                S.probas[name] = best.predict_proba(S.Xte_t)[:, 1]
                S.params[name] = params
                bar.progress((i + 1) / len(chosen))
        if "models" in S and S.models:
            rows = [{"Model": n, **metrics_at(S.yte, p)} for n, p in S.probas.items()]
            st.subheader("Comparison (test set, threshold 0.5)")
            st.dataframe(pd.DataFrame(rows).set_index("Model").round(3)
                         .sort_values("AUC", ascending=False), use_container_width=True)
            st.json(S.params)

# ---------------- 4. EVALUATION ----------------
with tabs[3]:
    if "models" not in S or not S.models:
        st.info("Train models in tab 3 first.")
    else:
        thr = st.slider("Decision threshold", 0.01, 0.99, 0.50, 0.01)
        credit = st.checkbox("Credit-scoring case: show Gini and KS", value=False)
        rows = [{"Model": n, **metrics_at(S.yte, p, thr)} for n, p in S.probas.items()]
        tbl = pd.DataFrame(rows).set_index("Model")
        cols = ["AUC", "PR-AUC", "Precision", "Recall", "F1"] + (["Gini", "KS"] if credit else [])
        st.dataframe(tbl[cols].round(3), use_container_width=True)

        sel = st.selectbox("Model for detailed view", list(S.models))
        m = tbl.loc[sel]
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{m['Precision']:.3f}")
        c2.metric("Recall", f"{m['Recall']:.3f}")
        c3.metric("F1", f"{m['F1']:.3f}")

        a, b = st.columns(2)
        fig = go.Figure()
        for n, p in S.probas.items():
            fpr, tpr, _ = roc_curve(S.yte, p)
            fig.add_scatter(x=fpr, y=tpr, name=f"{n} (AUC {tbl.loc[n,'AUC']:.3f})")
        fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), showlegend=False)
        fig.update_layout(title="ROC curves", xaxis_title="False positive rate",
                          yaxis_title="True positive rate")
        a.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        for n, p in S.probas.items():
            pr, rc, _ = precision_recall_curve(S.yte, p)
            fig.add_scatter(x=rc, y=pr, name=n)
        pr_sel, rc_sel = m["Precision"], m["Recall"]
        fig.add_scatter(x=[rc_sel], y=[pr_sel], mode="markers",
                        marker=dict(size=12, color="black"), name=f"{sel} @ threshold")
        fig.update_layout(title="Precision-Recall curves", xaxis_title="Recall", yaxis_title="Precision")
        b.plotly_chart(fig, use_container_width=True)

        pred = (S.probas[sel] >= thr).astype(int)
        cm = confusion_matrix(S.yte, pred)
        st.plotly_chart(px.imshow(cm, text_auto=True, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
                                  color_continuous_scale="Blues", title=f"Confusion matrix: {sel}"),
                        use_container_width=True)

# ---------------- 5. XAI ----------------
def cached(key, fn):
    S.setdefault("xai_cache", {})
    if key not in S.xai_cache:
        S.xai_cache[key] = fn()
    return S.xai_cache[key]

def show_fig():
    st.pyplot(plt.gcf(), clear_figure=True)
    plt.close("all")

def hbar(series, title, n=15):
    s = series.head(n)[::-1]
    return px.bar(x=s.values, y=s.index, orientation="h", title=title,
                  labels={"x": "", "y": ""})

def individual_form(Xraw):
    """Form for a fictional individual; values apply only on submit."""
    vals = {}
    with st.form("indiv"):
        cols = st.columns(3)
        for i, c in enumerate(Xraw.columns):
            s = Xraw[c]
            with cols[i % 3]:
                if pd.api.types.is_integer_dtype(s):
                    vals[c] = st.number_input(c, int(s.min()), int(s.max()), int(s.median()),
                                              step=1, key=f"ind_{c}")
                elif pd.api.types.is_numeric_dtype(s):
                    vals[c] = st.number_input(c, float(s.min()), float(s.max()),
                                              float(s.median()), key=f"ind_{c}")
                else:
                    opts = sorted(s.dropna().astype(str).unique())
                    if s.isna().any():
                        opts.append("(missing)")
                    v = st.selectbox(c, opts, key=f"ind_{c}")
                    vals[c] = np.nan if v == "(missing)" else v
        st.form_submit_button("Apply values")
    return pd.DataFrame([vals]).astype(Xraw.dtypes.to_dict())

with tabs[4]:
    if "models" not in S or not S.models:
        st.info("Train models in tab 3 first.")
    else:
        name = st.selectbox("Model to explain", list(S.models), key="xai_model")
        model = S.models[name]
        mid = id(model)
        cat_cols = list(S.prep.transformers_[1][2])
        slow = name == "kNN"

        # ===== GLOBAL =====
        st.header("Global explanations")
        a, b = st.columns(2)
        nat = native_importance(model, S.Xtr_t.columns)
        if nat is None:
            a.info("kNN has no native importance. Use permutation importance and SHAP.")
        else:
            kind = "coefficients" if hasattr(model, "coef_") else "impurity importance"
            a.plotly_chart(hbar(nat, f"Native importance ({kind})"), use_container_width=True)
            if hasattr(model, "coef_"):
                a.caption("One-hot columns are not standardized, so coefficients are not "
                          "comparable across variable types. Prefer permutation importance.")

        with st.spinner("Permutation importance..."):
            perm = cached(("perm", mid), lambda: perm_importance(
                model, S.Xte_t, S.yte, n_max=1000 if slow else 3000))
        b.plotly_chart(hbar(perm, "Permutation importance (drop in AUC)"), use_container_width=True)

        n_rows = st.slider("Test rows used for SHAP", 50, 1000, 100 if slow else 500, 50)
        with st.spinner("Computing SHAP values..."):
            sv_g = cached(("shap", mid, n_rows),
                          lambda: get_shap(model, S.Xtr_t, S.Xte_t.iloc[:n_rows]))
        st.subheader("SHAP summary (beeswarm)")
        shap.plots.beeswarm(sv_g, max_display=15, show=False)
        show_fig(7, 4.5)
        st.caption("Each dot is one individual. Horizontal position: push toward the positive "
                   "class (right) or away (left). Colour: feature value (red = high). "
                   "SHAP units are log-odds for Logistic/XGBoost and probability for tree "
                   "ensembles and kNN: compare rankings, not magnitudes, across models.")

        st.subheader("SHAP dependence plot")
        mean_abs = pd.Series(np.abs(sv_g.values).mean(0), index=sv_g.feature_names)
        num_cols = [c for c in sv_g.feature_names if S.Xtr_t[c].nunique() > 2]
        options = list(mean_abs[num_cols].nlargest(5).index) + list(mean_abs.nlargest(5).index)
        feat = st.selectbox("Variable", list(dict.fromkeys(options)))
        shap.plots.scatter(sv_g[:, feat], show=False)
        show_fig(7, 4.5)

        with st.expander("Sensitive-variable check (fairness)"):
            default = [c for c in ["age", "marital", "education", "job", "sex", "gender"]
                       if c in S.Xtr_raw.columns]
            sens = st.multiselect("Sensitive variables", list(S.Xtr_raw.columns), default=default)
            share = shap_by_variable(sv_g, cat_cols)
            st.write(f"Share of total SHAP importance carried by sensitive variables: "
                     f"**{share[share.index.isin(sens)].sum():.1f}%**")
            st.plotly_chart(hbar(share, "Share of SHAP importance by original variable (%)"),
                            use_container_width=True)

        # ===== LOCAL =====
        st.header("Local explanation (one individual)")
        thr = st.slider("Decision threshold", 0.01, 0.99, 0.50, 0.01, key="thr_xai")
        mode = st.radio("Individual", ["From the test set", "Custom individual"], horizontal=True)
        if mode == "From the test set":
            idx = int(st.number_input("Test row index", 0, len(S.Xte_t) - 1, 0, step=1))
            raw = S.Xte_raw.iloc[[idx]]
            st.caption(f"True label of this individual: {int(S.yte.iloc[idx])}")
        else:
            raw = individual_form(S.Xtr_raw)
        st.dataframe(raw.fillna("(missing)").astype(str).T.rename(columns=lambda _: "value"),
             use_container_width=True)

        x_t = S.prep.transform(raw)
        proba = float(model.predict_proba(x_t)[0, 1])
        c1, c2 = st.columns(2)
        c1.metric("Predicted probability (positive class)", f"{proba:.1%}")
        c2.metric(f"Decision at threshold {thr:.0%}", "POSITIVE" if proba >= thr else "NEGATIVE")

        with st.spinner("SHAP and LIME for this individual..."):
            sv_i = get_shap(model, S.Xtr_t, x_t)[0]
            lime_exp = cached(("lime", id(S.Xtr_t)), lambda: build_lime(S.Xtr_t))
            lime_s = lime_explain(model, lime_exp, S.Xtr_t.columns, x_t.iloc[0])

        st.success(summary_sentence(sv_i, proba, thr, cat_cols))

        a, b = st.columns(2)
        with a:
            st.subheader("SHAP waterfall")
            shap.plots.waterfall(sv_i, max_display=12, show=False)
            show_fig(7, 4.5)
        with b:
            st.subheader("LIME explanation")
            st.plotly_chart(hbar(lime_s.sort_values(key=np.abs, ascending=False),
                                 "LIME weights (top 10)", n=10), use_container_width=True)

        st.subheader("SHAP vs LIME")
        shap_s = pd.Series(sv_i.values, index=sv_i.feature_names)
        top = list(dict.fromkeys(list(shap_s.abs().nlargest(8).index) +
                                 list(lime_s.abs().nlargest(8).index)))
        cmp = pd.DataFrame({"SHAP": shap_s.reindex(top),
                            "LIME": lime_s.reindex(top).fillna(0)})
        cmp.index.name = "feature"
        st.plotly_chart(px.bar(cmp.reset_index().melt("feature"), x="value", y="feature",
                               color="variable", barmode="group", orientation="h"),
                        use_container_width=True)
        st.metric("Top-5 agreement (SHAP vs LIME)", f"{shap_lime_agreement(sv_i, lime_s):.0%}")
        st.caption("Scales differ between the two methods, so compare signs and rankings. "
                   "LIME shows only its top 10 features (others appear as 0).")