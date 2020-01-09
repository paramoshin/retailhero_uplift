from datetime import datetime
import pickle

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV


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

    y_train = (((train_is_treatment == 1) & (y_train == 1)) | ((train_is_treatment == 0) & (y_train == 0))).astype(int)
    y_valid = (((valid_is_treatment == 1) & (y_valid == 1)) | ((valid_is_treatment == 0) & (y_valid == 0))).astype(int)

    # params = {
    #     "learning_rate": [0.1, 0.01, 0.001],
    #     "gamma": [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
    #     "max_depth": [2, 4, 7, 10],
    #     "colsample_bytree": [0.3, 0.6, 0.8, 1.0],
    #     "subsample": [0.2, 0.4, 0.5, 0.6, 0.7],
    #     "reg_alpha": [0, 0.5, 1],
    #     "reg_lambda": [1, 1.5, 2, 3, 4.5],
    #     "min_child_weight": [1, 3, 5, 7],
    #     "n_estimators": [100, 250, 500, 1000]
    # }
    best_params = dict(
        subsample=0.6,
        reg_lambda=1.5,
        reg_alpha=0,
        n_estimators=500,
        min_child_weight=5,
        max_depth=4,
        learning_rate=0.01,
        gamma=0.1,
        colsample_bytree=0.6
    )
    clf = xgb.XGBClassifier(objective='binary:logistic', **best_params)
    # kfold = StratifiedKFold(n_splits=10, random_state=42)
    # xgb_rscv = RandomizedSearchCV(
    #     clf, param_distributions=params, scoring="roc_auc", cv=kfold, verbose=3, random_state=42, n_jobs=-1
    # )

    # xgb_rscv.fit(X_train, y_train)
    # print(f"best params: {xgb_rscv.best_params_}")
    # print(f"validation score: {xgb_rscv.score(X_valid, y_valid)}")

    X_train = pd.concat([X_train, X_valid], ignore_index=False)
    y_train = pd.concat([y_train, y_valid], ignore_index=False)

    model = clf.fit(X_train, y_train)

    dt = datetime.now().strftime("%Y-%m-%d_%HH-%MM")
    model_name = dt + "_" + str(model.__class__).split("'")[1].replace(".", "_")
    f_name = "../../models/class_transform/" + model_name

    pickle.dump(model, open(f_name + ".pkl", "wb"))
    # with open("../../models/class_transform/validation.csv", "a") as f:
    #     f.write(f"{model_name},{valid_uplift_score}\n")

    predict_test_proba = model.predict_proba(X_test)[:, 1]
    predict_test_uplift = 2 * predict_test_proba - 1
    df_submission = pd.DataFrame({'uplift': predict_test_uplift}, index=X_test.index)
    df_submission.to_csv(f'../../data/submissions/class_transform/{model_name}.csv')
