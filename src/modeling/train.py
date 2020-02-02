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
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from matplotlib import pyplot as plt
from mlflow import log_metric, log_param, log_artifact

from src.modeling.utils import *
from src.modeling.models import models

# 01.02 no best_params:
#   - level_1 - 0,09
#   - recency - 0,09
#   - frequency - 0,09
#   - recency, frequency - 0,0420
#   - recency, frequency, level_1 - 0,0420


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xgb")
    parser.add_argument("--recency", type=bool, default=False)
    parser.add_argument("--frequency", type=bool, default=False)
    parser.add_argument("--level_1", type=bool, default=False)
    parser.add_argument("--refit", type=bool, default=False)
    parser.add_argument("--use_best", type=bool, default=False)

    args = parser.parse_args()

    log_param("model", args.model)
    log_param("recency", args.recency)
    log_param("frequency", args.frequency)
    log_param("level_1", args.level_1)
    log_param("refit", args.refit)
    log_param("use_best", args.use_best)

    X_train, y_train, train_is_treatment, X_valid, y_valid, valid_is_treatment, X_test = read_train_test()
    X_train, y_train = join_train_validation(X_train, X_valid, y_train, y_valid)
    train_is_treatment = pd.concat([train_is_treatment, valid_is_treatment], ignore_index=False)
    folds = pd.read_csv("../../data/processed/folds.csv", index_col="client_id")

    if args.recency:
        recency = pd.read_csv("../../data/processed/recency.csv", index_col="client_id")
        X_train = X_train.join(recency)
    if args.frequency:
        frequency = pd.read_csv("../../data/processed/frequency.csv", index_col="client_id")
        X_train = X_train.join(frequency)
    if args.level_1:
        level_1 = pd.read_csv("../../data/processed/level_1.csv", index_col="client_id").drop(["Unnamed: 0"], axis=1)
        X_train = X_train.join(level_1)

    dt = datetime.now().strftime("%Y-%m-%d-%H-%M")

    metrics = {
        "control_acc": [],
        "treatment_acc": [],
        "control_auc": [],
        "treatment_auc": [],
        "uplift": []
    }
    control_best_params = {}
    treatment_best_params = {}

    if args.model == "xgb" and args.use_best:
        with open("../../models/control_xgb_best_params.json", "r") as f:
            control_best_params = json.load(f)
        with open("../../models/treatment_xgb_best_params.json", "r") as f:
            treatment_best_params = json.load(f)

    if args.model != "xgb":
        X_train.fillna(-999999, inplace=True)
        X_test.fillna(-999999, inplace=True)

    for i in range(5):
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

        if args.model == "xgb":
            clf_control = xgb.XGBClassifier(objective="binary:logistic", **control_best_params)\
                .fit(X_train_control, y_train_control)
            clf_treatment = xgb.XGBClassifier(objective="binary:logistic", **treatment_best_params)\
                .fit(X_train_treatment, y_train_treatment)
        else:
            clf_control = models[args.model].fit(X_train_control, y_train_control)
            clf_treatment = models[args.model].fit(X_train_treatment, y_train_treatment)

        treatment_proba = clf_treatment.predict_proba(test_data)[:, 1]
        control_proba = clf_control.predict_proba(test_data)[:, 1]
        uplift_prediction = treatment_proba - control_proba
        up_score = uplift_score(uplift_prediction, test_target, test_data_is_treatment)
        
        control_acc = clf_control.score(test_data, test_target)
        treatment_acc = clf_treatment.score(test_data, test_target)
        control_auc = roc_auc_score(test_target, control_proba)
        treatment_auc = roc_auc_score(test_target, treatment_proba)
        uplift = up_score

        log_metric("control-acc", control_acc)
        log_metric("treatment-acc", treatment_acc)
        log_metric("control-auc", control_auc)
        log_metric("treatment-auc", treatment_auc)
        log_metric("uplift", uplift)

        metrics["control_acc"].append(control_acc)
        metrics["treatment_acc"].append(treatment_acc)
        metrics["control_auc"].append(control_auc)
        metrics["treatment_auc"].append(treatment_auc)
        metrics["uplift"].append(uplift)
    
        folder = Path(f"../../models/two_models/folds/{dt}")
        folder.mkdir(parents=True, exist_ok=True)
        model_name = args.model + f"_fold_{i}"
        joblib.dump(clf_control, (folder / Path(model_name + "_control.pkl")).resolve())
        joblib.dump(clf_treatment, (folder / Path(model_name + "_treatment.pkl")).resolve())

        log_artifact((folder / Path(model_name + "_control.pkl")).resolve(), f"control-clf-fold-{i}")
        log_artifact((folder / Path(model_name + "_treatment.pkl")).resolve(), f"treatment-clf-fold-{i}")
    
    log_param("clf-control-params", clf_control.get_params())
    log_param("clf-treatment-params", clf_treatment.get_params())

    # print(metrics)
    avg_metrics = dict(zip(metrics.keys(), list(map(np.mean, metrics.values()))))
    # print(avg_metrics)
    for k, v in avg_metrics.items():
        log_metric("average-" + k, v)

    if args.refit:
        X_train, y_train, train_is_treatment, X_valid, y_valid, valid_is_treatment, X_test = read_train_test()
        X_train, y_train = join_train_validation(X_train, X_valid, y_train, y_valid)
        train_is_treatment = pd.concat([train_is_treatment, valid_is_treatment], ignore_index=False)
        X_train_control, X_train_treatment, y_train_control, y_train_treatment = split_control_treatment(
            X_train, y_train, train_is_treatment
        )

        if args.model == "xgb":
            clf_control = xgb.XGBClassifier(objective="binary:logistic", **control_best_params)\
                .fit(X_train_control, y_train_control)
            clf_treatment = xgb.XGBClassifier(objective="binary:logistic", **treatment_best_params)\
                .fit(X_train_treatment, y_train_treatment)

        fig, ax = plt.subplots(figsize=(20, 16))
        xgb.plot_importance(clf_control, ax=ax)
        plt.savefig("../../data/xgb_control_feature_importance.png", max_num_features=30)
        log_artifact("../../data/xgb_control_feature_importance.png", "contol-feature-importance")
        fig, ax = plt.subplots(figsize=(20, 16))
        xgb.plot_importance(clf_treatment, ax=ax)
        plt.savefig("../../data/xgb_treatment_feature_importance.png", max_num_features=30)
        log_artifact("../../data/xgb_treatment_feature_importance.png", "treatment-feature-importance")

        joblib.dump(clf_control, Path(f"../../models/two_models/{args.model}_refit_control.pkl").resolve())
        joblib.dump(clf_treatment, Path(f"../../models/two_models/{args.model}_refit_treatment.pkl").resolve())
        log_artifact(Path(f"../../models/two_models/{args.model}_refit_control.pkl").resolve(), f"control-clf-fold-{i}")
        log_artifact(Path(f"../../models/two_models/{args.model}_refit_treatment.pkl").resolve(), f"treatment-clf-fold-{i}")

        treatment_proba = clf_treatment.predict_proba(X_test)[:, 1]
        control_proba = clf_control.predict_proba(X_test)[:, 1]
        uplift_prediction = treatment_proba - control_proba

        df_submission = pd.DataFrame({'uplift': uplift_prediction}, index=X_test.index)
        submission_folder = Path(f"../../data/submissions/two_models/refit/")
        submission_folder.mkdir(parents=True, exist_ok=True)
        df_submission.to_csv(f"{submission_folder}/submission_{dt}.csv")
        log_artifact(f"{submission_folder}/submission_{dt}.csv", "submission")
