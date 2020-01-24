__all__ = ["uplift_score", "read_train_test", "join_train_validation", "split_control_treatment"]

import pandas as pd
import numpy as np


def read_train_test():
    X_train = pd.read_csv(
        "../../data/processed/two_models/X_train.csv", index_col="client_id"
    )
    y_train = pd.read_csv(
        "../../data/processed/two_models/y_train.csv",
        header=None,
        names=["client_id", "target"],
        index_col="client_id"
    )["target"]
    train_is_treatment = pd.read_csv(
        "../../data/processed/two_models/X_train_is_treatment.csv",
        header=None,
        names=["client_id", "is_treatment"],
        index_col="client_id"
    )["is_treatment"]

    X_valid = pd.read_csv("../../data/processed/two_models/X_valid.csv", index_col="client_id")
    y_valid = pd.read_csv(
        "../../data/processed/two_models/y_valid.csv",
        header=None,
        names=["client_id", "target"],
        index_col="client_id"
    )["target"]
    valid_is_treatment = pd.read_csv(
        "../../data/processed/two_models/X_valid_is_treatment.csv",
        header=None,
        names=["client_id", "is_treatment"],
        index_col="client_id"
    )["is_treatment"]

    X_test = pd.read_csv("../../data/processed/two_models/X_test.csv", index_col="client_id")

    return X_train, y_train, train_is_treatment, X_valid, y_valid, valid_is_treatment, X_test


def join_train_validation(X_train, X_valid, y_train, y_valid):
    X_train = pd.concat([X_train, X_valid], ignore_index=False)
    y_train = pd.concat([y_train, y_valid], ignore_index=False)
    return X_train, y_train


def split_control_treatment(X, y, is_treatment):
    X_control = X[is_treatment == 0]
    X_treatment = X[is_treatment == 1]
    y_control = y[is_treatment == 0]
    y_treatment = y[is_treatment == 1]
    return X_control, X_treatment, y_control, y_treatment


def uplift_score(prediction, treatment, target, rate=0.3):
    """
    Подсчет Uplift Score
    """
    order = np.argsort(-prediction)
    treatment_n = int((treatment == 1).sum() * rate)
    treatment_p = target[order][treatment[order] == 1][:treatment_n].mean()
    control_n = int((treatment == 0).sum() * rate)
    control_p = target[order][treatment[order] == 0][:control_n].mean()
    score = treatment_p - control_p
    return score
