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

from src.modeling.utils import *
from src.modeling.models import models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', default='xgb')
    arg('--refit', type=bool, default=False)

    args = parser.parse_args()

    X_train, y_train, train_is_treatment, X_valid, y_valid, valid_is_treatment, X_test = read_train_test()
    X_train, y_train = join_train_validation(X_train, X_valid, y_train, y_valid)
    train_is_treatment = pd.concat([train_is_treatment, valid_is_treatment], ignore_index=False)
    folds = pd.read_csv("../../data/processed/folds.csv", index_col="client_id")
    dt = datetime.now().strftime("%Y-%m-%d-%H-%M")

    if args.model != "xgb":
        X_train.fillna(-999, inplace=True)
        X_test.fillna(-999, inplace=True)
        if args.model in ("logreg", "knn"):
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index)
            X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index)

    metrics = {
        "control_acc": [],
        "treatment_acc": [],
        "control_auc": [],
        "treatment_auc": [],
        "uplift": []
    }

    xgb_best_params = {
        "subsample": 0.7,
        "reg_lambda": 1.0,
        "reg_alpha": 0,
        "n_estimators": 1000,
        "min_child_weight": 7,
        "max_depth": 4,
        "learning_rate": 0.01,
        "gamma": 0.3,
        "colsample_bytree": 0.5
    }
    with open("../../models/control_xgb_best_params.json", "r") as f:
        control_best_params = json.load(f)
    with open("../../models/treatment_xgb_best_params.json", "r") as f:
        treatment_best_params = json.load(f)

    # for i in range(5):
    #     print(f"Fold {i + 1}")
    #     test_idx = folds[folds["fold"] == i].index
    #     train_idx = folds[folds["fold"] != i].index
    #
    #     train_data = X_train.loc[train_idx]
    #     train_target = y_train.loc[train_idx]
    #     train_data_is_treatment = train_is_treatment.loc[train_idx]
    #
    #     test_data = X_train.loc[test_idx]
    #     test_target = y_train.loc[test_idx]
    #     test_data_is_treatment = train_is_treatment.loc[test_idx]
    #
    #     X_train_control, X_train_treatment, y_train_control, y_train_treatment = split_control_treatment(
    #         train_data, train_target, train_data_is_treatment
    #     )
    #     X_valid_control, X_valid_treatment, y_valid_control, y_valid_treatment = split_control_treatment(
    #         test_data, test_target, test_data_is_treatment
    #     )
    #
    #     if args.model == "xgb":
    #         clf_control = xgb.XGBClassifier(objective="binary:logistic", **control_best_params)\
    #             .fit(X_train_control, y_train_control)
    #         clf_treatment = xgb.XGBClassifier(objective="binary:logistic", **treatment_best_params)\
    #             .fit(X_train_treatment, y_train_treatment)
    #     else:
    #         clf_control = models[args.model].fit(X_train_control, y_train_control)
    #         clf_treatment = models[args.model].fit(X_train_treatment, y_train_treatment)
    #
    #     treatment_proba = clf_treatment.predict_proba(test_data)[:, 1]
    #     control_proba = clf_control.predict_proba(test_data)[:, 1]
    #     uplift_prediction = treatment_proba - control_proba
    #     up_score = uplift_score(uplift_prediction, test_target, test_data_is_treatment)
    #
    #     metrics["control_acc"].append(clf_control.score(test_data, test_target))
    #     metrics["treatment_acc"].append(clf_treatment.score(test_data, test_target))
    #     metrics["control_auc"].append(roc_auc_score(test_target, control_proba))
    #     metrics["treatment_auc"].append(roc_auc_score(test_target, treatment_proba))
    #     metrics["uplift"].append(up_score)
    #
    #     folder = Path(f"../../models/two_models/folds/{dt}")
    #     folder.mkdir(parents=True, exist_ok=True)
    #     model_name = args.model + f"_fold_{i}"
    #     joblib.dump(clf_control, (folder / Path(model_name + "_control.pkl")).resolve())
    #     joblib.dump(clf_treatment, (folder / Path(model_name + "_treatment.pkl")).resolve())
    #
    # print(metrics)
    # print(dict(zip(metrics.keys(), list(map(np.mean, metrics.values())))))

    if args.refit:
        X_train, y_train, train_is_treatment, X_valid, y_valid, valid_is_treatment, X_test = read_train_test()
        X_train, y_train = join_train_validation(X_train, X_valid, y_train, y_valid)
        train_is_treatment = pd.concat([train_is_treatment, valid_is_treatment], ignore_index=False)
        X_train_control, X_train_treatment, y_train_control, y_train_treatment = split_control_treatment(
            X_train, y_train, train_is_treatment
        )

        clf_control = xgb.XGBClassifier(objective="binary:logistic", **xgb_best_params).fit(X_train, y_train)
        clf_treatment = xgb.XGBClassifier(objective="binary:logistic", **xgb_best_params).fit(X_train, y_train)

        treatment_proba = clf_treatment.predict_proba(X_test)[:, 1]
        control_proba = clf_control.predict_proba(X_test)[:, 1]
        uplift_prediction = treatment_proba - control_proba

        df_submission = pd.DataFrame({'uplift': uplift_prediction}, index=X_test.index)
        submission_folder = Path(f"../../data/submissions/two_models/refit/")
        submission_folder.mkdir(parents=True, exist_ok=True)
        df_submission.to_csv(f'{submission_folder}/submission_{dt}.csv')
