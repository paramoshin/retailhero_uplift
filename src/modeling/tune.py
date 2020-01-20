import json
import sys
from pathlib import Path
p = str(Path(".").resolve().parent.parent)
sys.path.extend([p])

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.base import clone

from src.modeling.read_data import *
from src.modeling.utils import uplift_score
from src.modeling.models import models

if __name__ == "__main__":
    space = {
        "learning_rate": [0.2, 0.1, 0.05, 0.01],
        "gamma": [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
        "max_depth": list(range(3, 12, 2)), # can narrow it down later
        "colsample_bytree": list([i / 10.0 for i in range(6, 10)]),
        "subsample": list([i / 10.0 for i in range(6, 10)]),
        "reg_alpha": [1e-5, 1e-2, 0.1, 1, 100],
        "reg_lambda": [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        "min_child_weight": list(range(1, 10, 2)),
        "n_estimators": [100, 250, 500, 1000]
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=142)
    rs_clf = RandomizedSearchCV(
        xgb.XGBClassifier(objective="binary:logistic"),
        param_distributions=space,
        scoring="logloss",
        cv=skf,
        n_iter=20,
        verbose=3,
        random_state=142,
        n_jobs=-1
    )
    X_train, y_train, train_is_treatment, X_valid, y_valid, valid_is_treatment, X_test = read_train_test()
    X_train, y_train = join_train_validation(X_train, X_valid, y_train, y_valid)
    train_is_treatment = pd.concat([train_is_treatment, valid_is_treatment], ignore_index=False)
    X_train_control, X_train_treatment, y_train_control, y_train_treatment = split_control_treatment(
        X_train, y_train, train_is_treatment
    )

    rs_control = clone(rs_clf).fit(X_train_control, y_train_control)
    control_best_params = rs_control.best_params_
    print(control_best_params)

    rs_treatment = clone(rs_clf).fit(X_train_treatment, y_train_treatment)
    treatment_best_params = rs_treatment.best_params_
    print(treatment_best_params)

    with open("../../models/control_xgb_best_params.json", "w") as f:
        json.dump(control_best_params, f)

    with open("../../models/treatment_xgb_best_params.json", "w") as f:
        json.dump(treatment_best_params, f)
