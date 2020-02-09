import json
import sys
from pathlib import Path

p = str(Path(".").resolve().parent.parent)
sys.path.extend([p])

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.base import clone

from src.modeling.utils import *
from src.modeling.models import models


if __name__ == "__main__":
    control_best_params = {
        "max_depth": 2,
        "min_child_weight": 6,
        "gamma": 1.2,
        "colsample_bytree": 0.8,
        "subsample": 0.9,
        "reg_alpha": 60,
        "learning_rate": 0.05,
        "n_estimators": 750,
        "min_child_weight": 15,
        "reg_lambda": 2.2,
    }
    treatment_best_params = {
        "max_depth": 3,
        "min_child_weight": 5,
        "gamma": 0,
        "colsample_bytree": 0.75,
        "subsample": 0.9,
        "reg_alpha": 100,
        "learning_rate": 0.1,
        "n_estimators": 400,
        "min_child_weight": 2,
        "reg_lambda": 1.2,
    }
    # control_space = {
    # "learning_rate": [0.01, 0.05, 0.1, 0.2],
    # "gamma": [1.2, 1.3, 1.5, 1.7, 2],
    # "max_depth": [2, 3, 4],
    # "colsample_bytree": [i / 100.0 for i in range(75, 90, 5)],
    # "subsample": [i / 100.0 for i in range(85, 100, 5)],
    # "reg_alpha": [i for i in range(50, 160, 10)],
    # "reg_lambda": [2.2, 2.3, 2.4, 2.5],
    # "min_child_weight": [10, 11, 12, 13, 14, 15, 16],
    # "n_estimators": [600, 650, 700, 750, 1500]
    # }
    # treatment_space = {
    # "learning_rate": [0.01, 0.05, 0.1, 0.2],
    # "gamma": [0, 0.01, 0.03, 0.05],
    # "max_depth": [2, 3, 4],
    # "colsample_bytree": [i / 100.0 for i in range(75, 90, 5)],
    # "subsample": [i / 100.0 for i in range(85, 100, 5)],
    # "reg_alpha": [i for i in range(50, 160, 10)],
    # "reg_lambda": [1.1, 1.2, 1.3, 1.4],
    # "min_child_weight": [1, 2, 3, 4, 10, 15],
    # "n_estimators": [400, 450, 500, 550, 600, 650, 1500]
    # }
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # rs_clf = RandomizedSearchCV(
    #     xgb.XGBClassifier(objective="binary:logistic"),
    #     param_distributions=space,
    #     scoring="neg_log_loss",
    #     cv=skf,
    #     n_iter=20,
    #     verbose=3,
    #     random_state=142,
    #     n_jobs=-1
    # )
    # clf_control = xgb.XGBClassifier(objective="binary:logistic", **control_best_params)
    # clf_treatment = xgb.XGBClassifier(objective="binary:logistic", **treatment_best_params)
    # gs_clf_control = GridSearchCV(
    #     clf_control,
    #     param_grid=control_space,
    #     cv=skf,
    #     scoring="roc_auc",
    #     n_jobs=-1,
    #     verbose=2
    # )
    # gs_clf_treatment = GridSearchCV(
    #     clf_treatment,
    #     param_grid=treatment_space,
    #     cv=skf,
    #     scoring="roc_auc",
    #     n_jobs=-1,
    #     verbose=2
    # )
    # X_train, y_train, train_is_treatment, X_valid, y_valid, valid_is_treatment, X_test = read_train_test()
    # X_train, y_train = join_train_validation(X_train, X_valid, y_train, y_valid)
    # train_is_treatment = pd.concat([train_is_treatment, valid_is_treatment], ignore_index=False)
    # X_train_control, X_train_treatment, y_train_control, y_train_treatment = split_control_treatment(
    #     X_train, y_train, train_is_treatment
    # )

    # rs_control = gs_clf_control.fit(X_train_control, y_train_control)
    # control_best_params = rs_control.best_params_

    # rs_treatment = gs_clf_treatment.fit(X_train_treatment, y_train_treatment)
    # treatment_best_params = rs_treatment.best_params_

    # print(f"control best params: {control_best_params}")
    # print(f"control best score: {rs_control.best_score_}\n")

    # print(f"treatment: {treatment_best_params}")
    # print(f"treatment best score: {rs_treatment.best_score_}")

    with open("../../models/control_xgb_best_params.json", "w") as f:
        json.dump(control_best_params, f)

    with open("../../models/treatment_xgb_best_params.json", "w") as f:
        json.dump(treatment_best_params, f)
