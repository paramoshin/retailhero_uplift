import sys
from pathlib import Path

p = str(Path(".").resolve().parent.parent)
sys.path.extend([p])

import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
import eli5

from src.modeling.utils import *


if __name__ == "__main__":

    (
        X_train,
        y_train,
        train_is_treatment,
        X_valid,
        y_valid,
        valid_is_treatment,
        X_test,
    ) = read_train_test()

    recency = pd.read_csv("../../data/processed/recency.csv", index_col="client_id")
    X_train = X_train.join(recency)
    X_valid = X_valid.join(recency)

    frequency = pd.read_csv("../../data/processed/frequency.csv", index_col="client_id")
    X_train = X_train.join(frequency)
    X_valid = X_valid.join(frequency)

    level_1 = pd.read_csv("../../data/processed/level_1.csv", index_col="client_id").drop(
        ["Unnamed: 0"], axis=1
    )
    X_train = X_train.join(level_1)
    X_valid = X_valid.join(level_1)

    lda = pd.read_csv("../../data/processed/bucket_types.csv", index_col=["client_id"])
    lda.columns = [f"lda_{x}" for x in lda.columns]
    X_train = X_train.join(lda)
    X_valid = X_valid.join(lda)

    w2v = pd.read_csv("../../data/processed/w2v_repr.csv", index_col=["client_id"])
    w2v.columns = [f"w2v_{x}" for x in w2v.columns]
    X_train = X_train.join(w2v)
    X_valid = X_valid.join(w2v)

    model = xgb.XGBClassifier(n_estimators=400, n_jobs=-1, random_state=42)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_metric="roc_auc",
        verbose=3,
        early_stopping_rounds=100,
    )
    perm = eli5.sklearn.PermutationImportance(model, random_state=42).fit(X_train, y_train)
    with open("../../data/eli5_top_50_features.html", "w") as f:
        f.write(eli5.show_weights(perm, top=50, feature_names=X_train.columns))

    top_features = [i for i in eli5.formatters.as_dataframe.explain_weights_df(model).feature][:50]
    print("top features:", top_features)
