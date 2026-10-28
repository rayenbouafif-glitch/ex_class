from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessor(X, num_impute="median", scale=True):
    num = X.select_dtypes("number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy=num_impute)),
        ("sc", StandardScaler() if scale else "passthrough"),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ct = ColumnTransformer(
        [("num", num_pipe, num), ("cat", cat_pipe, cat)],
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    return ct