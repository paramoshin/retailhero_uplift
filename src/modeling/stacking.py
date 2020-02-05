import argparse
from datetime import datetime
import json
import sys
from pathlib import Path
p = str(Path(".").resolve().parent.parent)
sys.path.extend([p])

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from mlflow import log_metric, log_param, log_artifact

from src.modeling.utils import *
from src.modeling.models import models


def fit_(model, X_train_control, X_train_treatment, y_train_control, y_train_treatment, X_test):
    clf_control = clone(model).fit(X_train_control, y_train_control)
    clf_treatment = clone(model).fit(X_train_treatment, y_train_treatment)
    treatment_proba = clf_treatment.predict_proba(X_test)[:, 1]
    control_proba = clf_control.predict_proba(X_test)[:, 1]
    uplift_prediction = treatment_proba - control_proba
    return uplift_prediction

def get_cv_score(model, fodls, X_train, y_train, train_is_treatment):
    control_acc = []
    treatment_acc = []
    control_auc = []
    treatment_auc = []
    uplift = []
    for i in range(folds["fold"].nunique()):
        print(f"Fold {i + 1}")
        test_idx = folds[folds["fold"] == i].index
        train_idx = folds[folds["fold"] != i].index
    
        train_data = X_train.loc[train_idx]
        train_target = y_train.loc[train_idx]
        train_data_is_treatment = train_is_treatment.loc[train_idx]
    
        test_data = X_train.loc[test_idx]
        test_target = y_train.loc[test_idx]
        test_data_is_treatment = train_is_treatment.loc[test_idx]
    
        X_train_control, X_train_treatment, y_train_control, y_train_treatment = split_control_treatment(
            train_data, train_target, train_data_is_treatment
        )
        X_valid_control, X_valid_treatment, y_valid_control, y_valid_treatment = split_control_treatment(
            test_data, test_target, test_data_is_treatment
        )
        clf_control = model.fit(X_train_control, y_train_control)
        clf_treatment = model.fit(X_train_treatment, y_train_treatment)

        treatment_proba = clf_treatment.predict_proba(test_data)[:, 1]
        control_proba = clf_control.predict_proba(test_data)[:, 1]
        uplift_prediction = treatment_proba - control_proba
        up_score = uplift_score(uplift_prediction, test_target, test_data_is_treatment)
        
        control_acc.append(clf_control.score(test_data, test_target))
        treatment_acc.append(clf_treatment.score(test_data, test_target))
        control_auc.append(roc_auc_score(test_target, control_proba))
        treatment_auc.append(roc_auc_score(test_target, treatment_proba))
        uplift.append(up_score)

    print(f"control_acc: {control_acc}")
    print(f"treatment_acc: {treatment_acc}")
    print(f"control_auc: {control_auc}")
    print(f"treatment_auc: {treatment_auc}")
    print(f"uplift: {uplift}")

    print(f"Average control_acc: {np.mean(control_acc)}")
    print(f"Average treatment_acc: {np.mean(treatment_acc)}")
    print(f"Average control_auc: {np.mean(control_auc)}")
    print(f"Average treatment_auc: {np.mean(treatment_auc)}")
    print(f"Average uplift: {np.mean(uplift)}")
    return control_acc, treatment_acc, control_auc, treatment_auc, uplift

if __name__ == "__main__":
    X_train, y_train, train_is_treatment, X_valid, y_valid, valid_is_treatment, X_test = read_train_test()
    X_train, y_train = join_train_validation(X_train, X_valid, y_train, y_valid)
    train_is_treatment = pd.concat([train_is_treatment, valid_is_treatment], ignore_index=False)
    folds = pd.read_csv("../../data/processed/folds.csv", index_col="client_id")

    base_features = [c for c in X_train.columns if not "last_month" in c]
    last_month_features = [c for c in X_train.columns if "last_month" in c]

    recency = pd.read_csv("../../data/processed/recency.csv", index_col="client_id")
    frequency = pd.read_csv("../../data/processed/frequency.csv", index_col="client_id")
    level_1 = pd.read_csv("../../data/processed/level_1.csv", index_col="client_id").drop(["Unnamed: 0"], axis=1)

    # 1 -> xgb, base
    # 2 -> xgb, base + last_month
    # 3 -> xgb, last_month + recency + frequency
    # 4 -> xgb, base + recency + frequency
    # 5 -> xgb, last_month + level_1
    # 6 -> LogReg, 1, 2, 3, 4, 5

    # 1

    X_train_control, X_train_treatment, y_train_control, y_train_treatment = split_control_treatment(
        X_train, y_train, train_is_treatment
    )
    steps = [
        (
            models["xgb"], 
            X_train_control[base_features], 
            X_train_treatment[base_features], 
            y_train_control, 
            y_train_treatment,
            X_test[base_features]
        ),
        (
            models["xgb"],
            X_train_control, 
            X_train_treatment, 
            y_train_control, 
            y_train_treatment, 
            X_test
        ),
        (
            models["lightgbm"], 
            X_train_control[last_month_features].join(recency).join(frequency).fillna(-99999), 
            X_train_treatment[last_month_features].join(recency).join(frequency).fillna(-99999), 
            y_train_control, 
            y_train_treatment,
            X_test[last_month_features].join(recency).join(frequency).fillna(-99999)
        ),
        (
            models["lightgbm"], 
            X_train_control[base_features].join(recency).join(frequency).fillna(-99999), 
            X_train_treatment[base_features].join(recency).join(frequency).fillna(-99999), 
            y_train_control, 
            y_train_treatment,
            X_test[base_features].join(recency).join(frequency).fillna(-99999)
        ),
        (
            models["extratrees"], 
            X_train_control[last_month_features].join(level_1).fillna(-99999), 
            X_train_treatment[last_month_features].join(level_1).fillna(-99999), 
            y_train_control, 
            y_train_treatment,
            X_test[last_month_features].join(level_1).fillna(-99999)
        ),
    ]
    uplift_preds = []
    for i, step in enumerate(steps):
        print(f"step {i + 1}")
        get_cv_score(step[0], fodls, *join_train_validation(*step[1:5]) train_is_treatment)
        uplift_preds.append(fit_(*step))
    uplift_preds = np.array(uplift_preds)
    print(uplift_preds.shape)
    print(pd.DataFrame(uplift_preds).corr())
    uplift_preds /= 5
    df_submission = pd.DataFrame({'uplift': uplift_prediction}, index=X_test.index)
    submission_folder = Path(f"../../data/submissions/")
    submission_folder.mkdir(parents=True, exist_ok=True)
    df_submission.to_csv(f"{submission_folder}/submission_stacking_avg.csv")
