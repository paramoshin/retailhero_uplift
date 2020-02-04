import argparse
import json
from functools import partial
import sys
from pathlib import Path
p = str(Path(".").resolve().parent.parent)
sys.path.extend([p])

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from src.modeling.utils import *


def objective(space, X_train, y_train):
    classifier = xgb.XGBClassifier(
        n_estimators = space["n_estimators"],
        max_depth = int(space["max_depth"]),
        learning_rate = space["learning_rate"],
        gamma = space["gamma"],
        min_child_weight = space["min_child_weight"],
        subsample = space["subsample"],
        colsample_bytree = space["colsample_bytree"],
        n_jobs=-1,
        random_state=42
    )
    classifier.fit(X_train, y_train)
    # Applying k-Fold Cross Validation
    logloss = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, scoring="neg_log_loss")
    cross_val_mean = logloss.mean()
    print("CrossValMean:", cross_val_mean)
    return {"loss": 1 - cross_val_mean, "status": STATUS_OK}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recency", type=bool, default=False)
    parser.add_argument("--frequency", type=bool, default=False)
    parser.add_argument("--level_1", type=bool, default=False)
    args = parser.parse_args()

    X_train, y_train, train_is_treatment, X_valid, y_valid, valid_is_treatment, X_test = read_train_test()
    X_train, y_train = join_train_validation(X_train, X_valid, y_train, y_valid)
    train_is_treatment = pd.concat([train_is_treatment, valid_is_treatment], ignore_index=False)

    if args.recency:
        recency = pd.read_csv("../../data/processed/recency.csv", index_col="client_id")
        X_train = X_train.join(recency)
    if args.frequency:
        frequency = pd.read_csv("../../data/processed/frequency.csv", index_col="client_id")
        X_train = X_train.join(frequency)
    if args.level_1:
        level_1 = pd.read_csv("../../data/processed/level_1.csv", index_col="client_id").drop(["Unnamed: 0"], axis=1)
        X_train = X_train.join(level_1)

    X_train_control, X_train_treatment, y_train_control, y_train_treatment = split_control_treatment(
        X_train, y_train, train_is_treatment
    )

    space = {
        "max_depth" : hp.choice("max_depth", range(5, 30, 1)),
        "learning_rate" : hp.quniform("learning_rate", 0.01, 0.5, 0.01),
        "n_estimators" : hp.choice("n_estimators", range(20, 1000, 5)),
        "gamma" : hp.quniform("gamma", 0, 0.50, 0.01),
        "min_child_weight" : hp.quniform("min_child_weight", 1, 10, 1),
        "subsample" : hp.quniform("subsample", 0.1, 1, 0.01),
        "colsample_bytree" : hp.quniform("colsample_bytree", 0.1, 1.0, 0.01),
        "reg_alpha" : hp.quniform("reg_alpha", 40,180,1),
        "reg_lambda" : hp.uniform("reg_lambda", 0,1),
    }

    # Optimize control:
    trials = Trials()
    best = fmin(
        fn=partial(objective, X_train=X_train_control, y_train=y_train_control),
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )
    print(f"Control best params: {best}")

    p = "../../data/models/hyperopt_"
    if args.recency:
        p += "_recency"
    if args.frequency:
        p += "_frequency"
    if args.level_1:
        p += "_level_1"

    with open (p + "_control.json", "w") as f:
        json.dump(best, f)

    # Optimize treatment:
    trials = Trials()
    best = fmin(
        fn=partial(objective, X_train=X_train_treatment, y_train=y_train_treatment),
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )
    print(f"Treatment best params: {best}")
    with open(p + "_treatment.json", "w") as f:
        json.dump(best, f)