import argparse
import json
from copy import copy
from functools import partial
import sys
from pathlib import Path
p = str(Path(".").resolve().parent.parent)
sys.path.extend([p])

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from src.modeling.utils import *


def objective(space, X_train, y_train):
    classifier = xgb.XGBClassifier(
        # n_estimators = int(space["n_estimators"]),
        # learning_rate = space["learning_rate"],
        # min_child_weight = space["min_child_weight"],
        # subsample = space["subsample"],
        **space,
        n_jobs=-1,
        random_state=42
    )
    classifier.fit(X_train, y_train)
    # Applying k-Fold Cross Validation
    loss = 1 - cross_val_score(
        estimator=classifier, 
        X=X_train, 
        y=y_train, 
        cv=StratifiedKFold(n_splits=5, random_state=42), 
        scoring="roc_auc", 
        n_jobs=-1
    ).mean()
    return {"loss": loss, "status": STATUS_OK}


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
        "learning_rate" : hp.quniform("learning_rate", 0.01, 0.5, 0.01),
        "n_estimators" : hp.choice("n_estimators", range(100, 1000, 10)),
        "min_child_weight" : hp.quniform("min_child_weight", 1, 10, 1),
        # "subsample" : hp.quniform("subsample", 0.1, 1, 0.01),
#        "reg_alpha" : hp.quniform("reg_alpha", 40,180,1),
#        "reg_lambda" : hp.uniform("reg_lambda", 0,1),
    }
    
    p = "../../models/hyperopt"
    if args.recency:
        p += "_recency"
    if args.frequency:
        p += "_frequency"
    if args.level_1:
        p += "_level_1"

    p_control = Path(p + "_control.json")
    p_treatment = Path(p + "_treatment.json")
    
    if p_control.exists():
        with open(p_control, "r") as f:
            d = json.load(f)
        space_control = copy(space)
        space_control.update(d)
    else:
        space_control = space

    space_treatment = {}
    if p_treatment.exists():
        with open(p_treatment, "r") as f:
            d = json.load(f)
        space_treatment =copy(space)
        space_treatment.update(d)
    else:
        space_treatment = {}
    
    # Optimize control:
    trials = Trials()
    best = fmin(
        fn=partial(objective, X_train=X_train_control, y_train=y_train_control),
        space=space_control,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
    )
    print(f"Control best params: {best}")

    best["learning_rate"] = float(best["learning_rate"])
    best["n_estimators"] = int(best["n_estimators"])
    best["min_child_weight"] = int(best["min_child_weight"])

    if p_control.exists():
        with open (p_control, "r") as f:
            d = json.load(f)
        best = d.update(best)
    with open(p_control, "w") as f:
        json.dump(best, f)

    # Optimize treatment:
    trials = Trials()
    best = fmin(
        fn=partial(objective, X_train=X_train_treatment, y_train=y_train_treatment),
        space=space_treatment,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
    )

    best["learning_rate"] = float(best["learning_rate"])
    best["n_estimators"] = int(best["n_estimators"])
    best["min_child_weight"] = int(best["min_child_weight"])

    print(f"Treatment best params: {best}")
    if p_treatment.exists():
        with open (p_treatment, "r") as f:
            d = json.load(f)
        best = d.update(best)
    with open(p_treatment, "w") as f:
        json.dump(best, f)
