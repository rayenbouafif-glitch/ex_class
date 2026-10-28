import pandas as pd
from ucimlrepo import fetch_ucirepo

def load_bank():
    ds = fetch_ucirepo(id=222)  # Bank Marketing (UCI)
    X = ds.data.features.copy()
    y = ds.data.targets.iloc[:, 0].map({"yes": 1, "no": 0}).astype(int)
    X = X.drop(columns=["duration"])  # leakage: only known after the call
    return X, y

def load_csv(file, target):
    df = pd.read_csv(file)
    return df.drop(columns=[target]), df[target]