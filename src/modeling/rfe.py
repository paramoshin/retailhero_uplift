import sys
from pathlib import Path
p = str(Path(".").resolve().parent.parent)
sys.path.extend([p])

import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFECV

from src.modeling.utils import *


if __name__ == "__main__":

    X_train, y_train, train_is_treatment, X_valid, y_valid, valid_is_treatment, X_test = read_train_test()
    X_train, y_train = join_train_validation(X_train, X_valid, y_train, y_valid)
    train_is_treatment = pd.concat([train_is_treatment, valid_is_treatment], ignore_index=False)

    recency = pd.read_csv("../../data/processed/recency.csv", index_col="client_id")
    X_train = X_train.join(recency)

    frequency = pd.read_csv("../../data/processed/frequency.csv", index_col="client_id")
    X_train = X_train.join(frequency)

    level_1 = pd.read_csv("../../data/processed/level_1.csv", index_col="client_id").drop(["Unnamed: 0"], axis=1)
    X_train = X_train.join(level_1)

    lda = pd.read_csv("../../data/processed/bucket_types.csv", index_col=["client_id"])
    lda.columns = [f"lda_{x}" for x in lda.columns]
    X_train = X_train.join(lda)

    w2v = pd.read_csv("../../data/processed/w2v_repr.csv", index_col=["client_id"])
    w2v.columns = [f"w2v_{x}" for x in w2v.columns]
    X_train = X_train.join(w2v)

    X_control, X_treatment, y_control, y_treatment = split_control_treatment(X_train, y_train, train_is_treatment)

    clf_control = xgb.XGBClassifier(random_state=42, n_jobs=-1)
    selector_control = RFECV(clf_control, step=1, min_features_to_select=1, cv=5, scoring='roc_auc_score', verbose=1)
    selector_control.fit(X_control, y_control)

    clf_treatment = xgb.XGBClassifier(random_state=42, n_jobs=-1)
    selector_treatment = RFECV(clf_treatment, step=1, min_features_to_select=1, cv=5, scoring='roc_auc_score', verbose=1)
    selector_treatment.fit(X_treatment, y_treatment)

    plt.figure()
    plt.title('XGB CV score vs No of Features (control)')
    plt.xlabel("Number of features selected")
    plt.ylabel("Roc AUC Score")
    plt.plot(range(1, len(selector_control.grid_scores_) + 1), selector_control.grid_scores_)
    plt.savefig("../../data/control_rfe_score.png")    

    plt.figure()
    plt.title('XGB CV score vs No of Features (treatment)')
    plt.xlabel("Number of features selected")
    plt.ylabel("Roc AUC Score")
    plt.plot(range(1, len(selector_treatment.grid_scores_) + 1), selector_treatment.grid_scores_)
    plt.savefig("../../data/treatment_rfe_score.png")    

    print("control", selector_control.get_support)
    print("treatment", selector_treatment.get_support)
