import math
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

def get_models(pos_weight):
    return {
        "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight="balanced"),
                                {"C": [0.01, 0.1, 1, 10]}),
        "Decision Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=0),
                          {"max_depth": [3, 5, 8, 12], "min_samples_leaf": [5, 20, 50]}),
        "kNN": (KNeighborsClassifier(), {"n_neighbors": [5, 15, 31]}),
        "Random Forest": (RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=0),
                          {"n_estimators": [100, 300], "max_depth": [6, 12, None]}),
        "XGBoost": (XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss", random_state=0),
                    {"n_estimators": [100, 300], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}),
    }

def tune(model, grid, Xtr, ytr, n_iter=6):
    size = math.prod(len(v) for v in grid.values())
    s = RandomizedSearchCV(model, grid, n_iter=min(n_iter, size), scoring="roc_auc",
                           cv=3, n_jobs=-1, random_state=0)
    s.fit(Xtr, ytr)
    return s.best_estimator_, s.best_params_