from datetime import datetime
import pickle

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV


if __name__ == "__main__":
    X_control_train = pd.read_csv(
        "../../data/processed/two_models/X_control_train.csv", index_col="client_id"
    )
    y_control_train = pd.read_csv(
        "../../data/processed/two_models/y_control_train.csv",
        header=None,
        names=["client_id", "target"],
        index_col="client_id"
    )["target"]

    X_treatment_train = pd.read_csv(
        "../../data/processed/two_models/X_treatment_train.csv", index_col="client_id"
    )
    y_treatment_train = pd.read_csv(
        "../../data/processed/two_models/y_treatment_train.csv",
        header=None,
        names=["client_id", "target"],
        index_col="client_id"
    )["target"]

    X_valid = pd.read_csv("../../data/processed/two_models/X_valid.csv", index_col="client_id")
    y_valid = pd.read_csv(
        "../../data/processed/two_models/y_valid.csv",
        header=None,
        names=["client_id", "target"],
        index_col="client_id"
    )["target"]

    valid_is_treatment = pd.read_csv(
        "../../data/processed/two_models/valid_is_treatment.csv",
        header=None,
        names=["client_id", "is_treatment"],
        index_col="client_id"
    )["is_treatment"]

    X_test = pd.read_csv("../../data/processed/two_models/X_test.csv", index_col="client_id")


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


    clf = xgb.XGBClassifier()
    parameters = {
        "learning_rate": [0.1, 0.01, 0.001],
        "gamma": [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
        "max_depth": [2, 4, 7, 10],
        "colsample_bytree": [0.3, 0.6, 0.8, 1.0],
        "subsample": [0.5, 0.6, 0.7, 0.8, 0.9],
        "reg_alpha": [0, 0.5, 1],
        "reg_lambda": [1, 1.5, 2, 3, 4.5],
        "min_child_weight": [1, 3, 5, 7],
        "n_estimators": [100, 250, 500, 1000]
    }

    xgb_rscv_control = RandomizedSearchCV(
        clone(clf), param_distributions=parameters, cv=3, verbose=3, random_state=42, n_jobs=-1
    )
    clf_control = xgb_rscv_control.fit(X_control_train, y_control_train)
    print(f"Score on control train set: {clf_control.score(X_control_train, y_control_train)}")
    print(f"Best params for control classifier: {xgb_rscv_control.best_params_ }, "
          f"best score: {xgb_rscv_control.best_score_}")
    print(
        f"Accuracy for control classifier on validataion set: "
        f"{clf_control.score(X_valid[valid_is_treatment == 0], y_valid[valid_is_treatment == 0])}"
    )
    
    X_treatment_train["control_pred"] = clf_control.predict_proba(X_treatment_train)[:, 1]
    xgb_rscv_treatment = RandomizedSearchCV(
        clone(clf), param_distributions=parameters, cv=3, verbose=3, random_state=42, n_jobs=-1
    )
    clf_treatment = xgb_rscv_treatment.fit(X_treatment_train, y_treatment_train)
    print(f"Score on treatment train set: {clf_treatment.score(X_treatment_train, y_treatment_train)}")
    print(f"Best params for treatment classifier: {xgb_rscv_treatment.best_params_ }, "
          f"best score: {xgb_rscv_treatment.best_score_}")

    print(
        f"Accuracy for treatment classifier on validation set: "
        f"{clf_control.score(X_valid[valid_is_treatment == 1], y_valid[valid_is_treatment == 1])}"
    )

    X_valid["control_pred"] = clf_control.predict_proba(X_valid)[:, 1]
    valid_uplift = clf_treatment.predict_proba(X_valid)[:, 1] - X_valid["control_pred"]
    valid_uplift_score = uplift_score(valid_uplift, valid_is_treatment, y_valid)
    print(f"Uplift score on validation = {valid_uplift_score}, "
          f"(baseline on validation = 0.05081166028966111, "
          f"difference = {valid_uplift_score - 0.0605}")

    dt = datetime.now().strftime("%Y-%m-%d_%HH-%MM")
    model_name = dt + "_" + str(clf.__class__).split("'")[1].replace(".", "_")
    f_name = "../../models/ddr/" + model_name

    pickle.dump(
        clf_control, open(f_name + "_control.pkl", "wb")
    )
    pickle.dump(
        clf_treatment, open(f_name + "_treatment.pkl", "wb")
    )
    with open("../../models/ddr/validation.csv", "a") as f:
        f.write(f"{model_name},{valid_uplift_score}\n")

    predict_test_control = clf_control.predict_proba(X_test)[:, 1]
    X_test["control_pred"] = predict_test_control
    predict_test_treatment = clf_treatment.predict_proba(X_test)[:, 1]
    predict_test_uplift = predict_test_treatment - predict_test_control
    df_submission = pd.DataFrame({'uplift': predict_test_uplift}, index=X_test.index)
    df_submission.to_csv(f'../../data/submissions/ddr/{model_name}.csv')
