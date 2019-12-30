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

    X_train = pd.concat([X_control_train, X_treatment_train], ignore_index=True)
    X_train["is_treatment"] = np.concatenate(
        [np.zeros_like(y_control_train), np.ones_like(y_treatment_train)]
    )
    X_train["target"] = pd.concat([y_control_train, y_treatment_train])
    X_train["Z"] = (
        (
            (X_train["is_treatment"] == 1) & (X_train["target"] == 1)
        ) | (
            (X_train["is_treatment"] == 0) & (X_train["target"] == 0)
        )
    ).astype(int)
    y_train = X_train["Z"]
    X_train = X_train.drop(["is_treatment", "target", "Z"], axis=1)

    X_valid["is_treatment"] = valid_is_treatment
    X_valid["target"] = y_valid
    X_valid["Z"] = (
        (
            (X_valid["is_treatment"] == 1) & (X_valid["target"] == 1)
        ) | (
            (X_valid["is_treatment"] == 0) & (X_valid["target"] == 0)
        )
    ).astype(int)
    y_valid = X_valid["Z"]
    X_valid = X_valid.drop(["is_treatment", "target", "Z"], axis=1)

    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train)
    print(f"Accuracy on validation set: {clf.score(X_valid, y_valid)}")

    validation_uplift = 2 * clf.predict_proba(X_valid) - 1
    valid_uplift_score = uplift_score(validation_uplift, valid_is_treatment, y_valid)
    print(f"Uplift score on validation set: {valid_uplift_score}"
          f"(baseline on validation = 0.05081166028966111, "
          f"difference = {valid_uplift_score - 0.05081166028966111}")

    dt = datetime.now().strftime("%Y-%m-%d_%HH-%MM")
    model_name = dt + "_" + str(clf.__class__).split("'")[1].replace(".", "_")
    f_name = "../../models/class_transform/" + model_name

    pickle.dump(
        clf, open(f_name + ".pkl", "wb")
    )
    with open("../../models/class_transform/validation.csv", "a") as f:
        f.write(f"{model_name},{valid_uplift_score}\n")

    predict_test_proba = clf.predict_proba(X_test)
    predict_test_uplift = 2 * predict_test_proba - 1
    df_submission = pd.DataFrame({'uplift': predict_test_uplift}, index=X_test.index)
    df_submission.to_csv(f'../../data/submissions/class_transform/{model_name}.csv')
