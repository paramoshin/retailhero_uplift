import argparse
import sys
from pathlib import Path
p = str(Path(".").resolve().parent.parent)
sys.path.extend([p])

import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from src.modeling.models import models
from src.utils import *

# TODO: add list of base features from xgboost feature_importance
BASE_FEATURES = [

]

def validation(X_train, y_train, train_is_treatment, folds, clf):
    metrics = {}
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

        clf_control = clone(clf).fit(X_train_control, y_train_control)
        clf_treatment = clone(clf).fit(X_train_treatment, y_train_treatment)

        treatment_proba = clf_treatment.predict_proba(test_data)[:, 1]
        control_proba = clf_control.predict_proba(test_data)[:, 1]
        uplift_prediction = rf_treatment_proba - rf_control_proba
        up_score = uplift_score(uplift_prediction, test_target, test_data_is_treatment)

        metrics["control_acc"].append(clf_control.score(test_data, test_target))
        metrics["treatment_acc"].append(clf_treatment.score(test_data, test_target))
        metrics["control_auc"].append(roc_auc_score(test_target, control_proba))
        metrics["treatment_auc"].append(roc_auc_score(test_target, treatment_proba))
        metrics["uplift"].append(up_score)

    return metrics, dict(zip(rf_metrics.keys(), list(map(np.mean, rf_metrics.values()))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str)
    args = paser.parse_args()

    features = args.feature_name.split(",")

    X_train, y_train, train_is_treatment, X_valid, y_valid, valid_is_treatment, X_test = read_train_test()
    X_train, y_train = join_train_validation(X_train, X_valid, y_train, y_valid)
    train_is_treatment = pd.concat([train_is_treatment, valid_is_treatment], ignore_index=False)
    folds = pd.read_csv("../../data/processed/folds.csv", index_col="client_id")

    X_train_base = X_train[BASE_FEATURES].fillna(-999999)
    X_train = X_train[BASE_FEATURES + features].fillna(-999999)

    rf = models["randomforest"]
    xtrees = models["extratrees"]

    # Check perfomance on base features:
    rf_metrics_base, rf_metrcis_base_mean = validation(X_train_base, y_train, train_is_treatment, folds, rf)
    xtrees_metrics_base, xtrees_metrcis_base_mean = validation(X_train_base, y_train, train_is_treatment, folds, xtrees)

    # Check perfomance on all features:
    rf_metrics, rf_metrcis_mean = validation(X_train, y_train, train_is_treatment, folds, rf)
    xtrees_metrics, xtrees_metrcis_mean = validation(X_train, y_train, train_is_treatment, folds, xtrees)

    print(
        "Random forest metrics difference on folds:", 
        {k: v2 - v1 for (k, v1), (k2, v2) in zip(rf_metrics_base.items(), rf_metrics.items())}
    )
    print(
        "Random forest difference on average metrics:", 
        {k: v2 - v1 for (k, v1), (k2, v2) iz zip(rf_metrcis_base_mean.itesm(), rf_metrcis_mean.items())}        
    )

    print(
        "Extratrees metrics difference on folds:", 
        {k: v2 - v1 for (k, v1), (k2, v2) in zip(xtrees_metrics_base.items(), xtrees_metrics.items())}
    )
    print(
        "Extratrees difference on average metrics:", 
        {k: v2 - v1 for (k, v1), (k2, v2) iz zip(xtrees_metrcis_mean.itesm(), xtrees_metrcis_base_mean.items())}        
    )
