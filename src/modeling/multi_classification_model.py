from datetime import datetime

import pandas as pd
import numpy as np
import xgboost as xgb


if __name__ == "__main__":
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

    X_train["new_target"] = 0
    X_train.loc[(train_is_treatment == 0) & (y_train == 1), "new_target"] = 1
    X_train.loc[(train_is_treatment == 1) & (y_train == 0), "new_target"] = 2
    X_train.loc[(train_is_treatment == 1) & (y_train == 1), "new_target"] = 3
    new_y_train = X_train["new_target"]
    X_train.drop("new_target", axis=1, inplace=True)

    X_valid["new_target"] = 0
    X_valid.loc[(valid_is_treatment == 0) & (y_valid == 1), "new_target"] = 1
    X_valid.loc[(valid_is_treatment == 1) & (y_valid == 0), "new_target"] = 2
    X_valid.loc[(valid_is_treatment == 1) & (y_valid == 1), "new_target"] = 3
    new_y_valid = X_valid["new_target"]
    X_valid.drop("new_target", axis=1, inplace=True)

    clf = xgb.XGBClassifier().fit(X_train, new_y_train)
    print(f"classifier score on validation set: {clf.score(X_valid, new_y_valid)}")
    class_probs = clf.predict_proba(X_valid)
    valid_uplift = class_probs[:, 0] + class_probs[:, 3] - class_probs[:, 1] - class_probs[:, 2]

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

    print(f"uplift score on validation: {uplift_score(valid_uplift, valid_is_treatment, y_valid)}")

    X_train = pd.concat([X_train, X_valid], ignore_index=False)
    y_train = pd.concat([new_y_train, new_y_valid], ignore_index=False)

    clf = xgb.XGBClassifier().fit(X_train, y_train)

    dt = datetime.now().strftime("%Y-%m-%d_%HH-%MM")
    model_name = dt + "_" + str(clf.__class__).split("'")[1].replace(".", "_")

    class_probs = clf.predict_proba(X_test)
    test_uplift = class_probs[:, 0] + class_probs[:, 3] - class_probs[:, 1] - class_probs[:, 2]
    df_submission = pd.DataFrame({'uplift': test_uplift}, index=X_test.index)
    df_submission.to_csv(f'../../data/submissions/multiclass/{model_name}.csv')
