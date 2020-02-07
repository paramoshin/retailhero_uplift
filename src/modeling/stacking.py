import argparse
from datetime import datetime
import json
from collections import namedtuple
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
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from mlflow import log_metric, log_param, log_artifact

from src.modeling.utils import *
from src.modeling.models import models


def fit_(model, X_train_control, X_train_treatment, y_train_control, y_train_treatment, X_test):
    clf_control = clone(model).fit(X_train_control, y_train_control)
    clf_treatment = clone(model).fit(X_train_treatment, y_train_treatment)
    treatment_proba = clf_treatment.predict_proba(X_test)[:, 1]
    control_proba = clf_control.predict_proba(X_test)[:, 1]
    return control_proba, treatment_proba

def get_cv_score(model, folds, X_train, y_train, train_is_treatment):
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
        clf_control = clone(model).fit(X_train_control, y_train_control)
        clf_treatment = clone(model).fit(X_train_treatment, y_train_treatment)

        treatment_proba = clf_treatment.predict_proba(test_data)[:, 1]
        control_proba = clf_control.predict_proba(test_data)[:, 1]
        uplift_prediction = treatment_proba - control_proba
        up_score = uplift_score(
            prediction=uplift_prediction, treatment=test_data_is_treatment.values, target=test_target.values
        )
        
        control_acc.append(clf_control.score(test_data, test_target))
        treatment_acc.append(clf_treatment.score(test_data, test_target))
        control_auc.append(roc_auc_score(test_target, control_proba))
        treatment_auc.append(roc_auc_score(test_target, treatment_proba))
        uplift.append(up_score)

    print(f"Average control_acc: {np.mean(control_acc)}")
    print(f"Average treatment_acc: {np.mean(treatment_acc)}")
    print(f"Average control_auc: {np.mean(control_auc)}")
    print(f"Average treatment_auc: {np.mean(treatment_auc)}")
    print(f"Average uplift: {np.mean(uplift)}")
    return control_acc, treatment_acc, control_auc, treatment_auc, uplift

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level_2", type=bool, default=False)
    parser.add_argument("--w1", type=float, default=0.5)
    parser.add_argument("--w2", type=float, default=0.5)

    args = parser.parse_args()

    X_train, y_train, train_is_treatment, X_valid, y_valid, valid_is_treatment, X_test = read_train_test()
    X_train, y_train = join_train_validation(X_train, X_valid, y_train, y_valid)
    train_is_treatment = pd.concat([train_is_treatment, valid_is_treatment], ignore_index=False)
    folds = pd.read_csv("../../data/processed/folds.csv", index_col="client_id")

    base_features = [c for c in X_train.columns if not "last_month" in c]
    last_month_features = [c for c in X_train.columns if "last_month" in c]

    recency = pd.read_csv("../../data/processed/recency.csv", index_col="client_id")
    frequency = pd.read_csv("../../data/processed/frequency.csv", index_col="client_id")
    level_1 = pd.read_csv("../../data/processed/level_1.csv", index_col="client_id").drop(["Unnamed: 0"], axis=1)

    # DATASET 1: base_features
    # DATASET 2: base_features + last_month_features
    # DATASET 3: base_features + last_month_features + recency + frequency
    # LEVEL 1:
    # RandmForest(Dataset1)
    # LogisticRegression(Dataset1)
    # LGBM(Dataset2)
    # GBDT(Dataset2)
    # Extratrees(Dataset3)
    # KNN(Dataset3)
    # LEVEL 2:
    # LogisticRegression
    # XGBoost
    # WeighedAverage

    X_train_control, X_train_treatment, y_train_control, y_train_treatment = split_control_treatment(
        X_train, y_train, train_is_treatment
    )
    scaler = StandardScaler()
    Step = namedtuple("Step", "model X_train_control X_train_treatment y_train_control y_train_treatment X_test")
    level_1_steps = [
        Step(
            clone(models["randomforest"]), 
            pd.DataFrame(
                X_train_control[base_features].fillna(-99999), 
                columns=base_features, 
                index=X_train_control.index
            ), 
            pd.DataFrame(
                X_train_treatment[base_features].fillna(-99999), 
                columns=base_features, 
                index=X_train_treatment.index
            ), 
            y_train_control, 
            y_train_treatment,
            pd.DataFrame(
                X_test[base_features].fillna(-99999), 
                columns=base_features,
                index=X_test.index
            )
        ),
        Step(
            clone(models["logreg"]),
            pd.DataFrame(
                scaler.fit_transform(X_train_control[base_features].fillna(-99999)),
                columns=base_features,
                index=X_train_control.index
            ), 
            pd.DataFrame(
                scaler.transform(X_train_treatment[base_features].fillna(-99999)),
                columns=base_features,
                index=X_train_treatment.index
            ), 
            y_train_control, 
            y_train_treatment, 
            pd.DataFrame(
                scaler.transform(X_test[base_features].fillna(-99999)),
                columns=base_features,
                index=X_test.index
            )
        ),
        Step(
            clone(models["lightgbm"]), 
            pd.DataFrame(
                X_train_control.fillna(-99999),
                columns=X_train_control.columns,
                index=X_train_control.index
            ), 
            pd.DataFrame(
                X_train_treatment.fillna(-99999),
                columns=X_train_treatment.columns,
                index=X_train_treatment.index
            ),
            y_train_control, 
            y_train_treatment,
            pd.DataFrame(
                X_test.fillna(-99999),
                columns=X_test.columns,
                index=X_test.index
            )
        ),
        Step(
            clone(models["gradientboosting"]), 
            pd.DataFrame(
                X_train_control.fillna(-99999), 
                columns=X_train_control.columns,
                index=X_train_control.index
            ),
            pd.DataFrame(
                X_train_treatment.fillna(-99999), 
                columns=X_train_treatment.columns,
                index=X_train_treatment.index
            ),
            y_train_control, 
            y_train_treatment,
            pd.DataFrame(
                X_test.fillna(-99999),
                columns=X_test.columns,
                index=X_test.index
            )
        ),
        Step(
            clone(models["extratrees"]), 
            pd.DataFrame(
                X_train_control[last_month_features].join(recency).join(frequency).fillna(-99999), 
                columns=last_month_features + recency.columns.tolist() + frequency.columns.tolist(),
                index=X_train_control.index
            ),
            pd.DataFrame(
                X_train_treatment[last_month_features].join(recency).join(frequency).fillna(-99999), 
                columns=last_month_features + recency.columns.tolist() + frequency.columns.tolist(),
                index=X_train_treatment.index
            ),
            y_train_control, 
            y_train_treatment,
            pd.DataFrame(
                X_test[last_month_features].join(recency).join(frequency).fillna(-99999),
                columns=last_month_features + recency.columns.tolist() + frequency.columns.tolist(),
                index=X_test.index
            )
        ),
        Step(
            clone(models["knn"]), 
            pd.DataFrame(
                scaler.fit_transform(
                    X_train_control[last_month_features]
                    .join(recency).join(frequency).fillna(-99999)
                ),
                columns=last_month_features + recency.columns.tolist() + frequency.columns.tolist(),
                index=X_train_control.index
            ),
            pd.DataFrame(
                scaler.transform(
                    X_train_treatment[last_month_features]
                    .join(recency).join(frequency).fillna(-99999)
                ),
                columns=last_month_features + recency.columns.tolist() + frequency.columns.tolist(),
                index=X_train_treatment.index
            ),
            y_train_control, 
            y_train_treatment,
                pd.DataFrame(
                scaler.transform(
                    X_test[last_month_features].join(recency).join(frequency).fillna(-99999)
                ),
                columns=last_month_features + recency.columns.tolist() + frequency.columns.tolist(),
                index=X_test.index
            )
        ),
    ]

    control_probas = [] 
    treatment_probas = []

    for i, step in enumerate(level_1_steps):
        print(f"------STEP {i + 1}------")
        get_cv_score(
            step.model, 
            folds, 
            *join_train_validation(
                step.X_train_control, step.X_train_treatment, step.y_train_control, step.y_train_treatment
            ), 
            train_is_treatment
        )
        control_proba, treatment_proba = fit_(*step)
        control_probas.append(control_proba)
        treatment_probas.append(treatment_proba)
        print("\n")
    
    if args.level_2:
        lr_control = clone(models["logreg"]).fit(np.array(control_probas).T, y_train_control)
        lr_treatment = clone(models["logreg"]).fit(np.array(treatment_probas).T, y_train_treatment)

        xgb_control = clone(models["xgb"]).fit(np.array(control_probas).T, y_train_control)
        xgb_treatment = clone(models["xgb"]).fit(np.array(control_probas).T, y_train_control)

        uplift_lr = lr_treatment.predict_proba(X_test)[:, 1] - lr_control.predict_proba(X_test)[:, 1]
        uplift_xgb = xgb_treatment.predict_proba(X_test)[:, 1] - xgb_control.predict_proba(X_test)[:, 1]

        uplift_preds_df = pd.DataFrame({"uplift_lr": uplift_lr, "uplift_xgb": uplift_xgb})
    else:
        for i, (treatment_proba, control_proba) in enumerate(zip(treatment_probas, control_probas)):
            d = {f"step_{i}": treatment_proba - control_proba}
        uplift_preds_df = pd.DataFrame(d)

    print(uplift_preds_df.corr())

    if args.level_2 and args.w1 != 0.5 and ars.w2 != 0.5:
        uplift_prediction = (w1 * uplift_lr) + (w2 * uplift_xgb)
    else:
        uplift_prediction = uplift_preds_df.mean(axis=1).values
        
    df_submission = pd.DataFrame({'uplift': uplift_prediction}, index=X_test.index)
    print(df_submission.head())

    submission_folder = Path(f"../../data/submissions/")
    submission_folder.mkdir(parents=True, exist_ok=True)

    f_name = "stacking_"
    if args.level_2:
        f_name += "level_2_"
        if args.w1 != 0.5 and args.w2 != 0.5:
            f_name += f"{str(w1).replace('.', '')}_{str(w2).replace('.', '')}"
    else:
        f_name += "level_1_avg"

    df_submission.to_csv(f"{submission_folder}/{f_name}.csv")
