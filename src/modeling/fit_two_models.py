from datetime import datetime
import pickle

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import stats
from sklearn.base import clone
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

    # best_params = {
    #     'colsample_bytree': 0.5701975341512912, 'learning_rate': 0.014065852851773964, 'max_depth': 7,
    #     'min_child_weight': 3, 'n_estimators': 458, 'subsample': 0.8849549260809972
    # }
    clf = xgb.XGBClassifier(objective='binary:logistic')
    param_dist = {
        'n_estimators': stats.randint(150, 1200),
        'learning_rate': [0.1, 0.01, 0.001],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
        'max_depth': [3, 4, 5, 6, 7, 8, 9],
        'colsample_bytree': [0.4, 0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        "reg_alpha": [0, 0.5, 1],
        "reg_lambda": [1, 1.5, 2, 3, 4.5],
        "gamma": [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
    }
    k_fold = StratifiedKFold(n_splits=5, random_state=42)
    rs_clf = RandomizedSearchCV(
        clf,
        cv=k_fold,
        param_distributions=param_dist,
        n_iter=20,
        scoring='f1',
        verbose=3,
        n_jobs=-1,
        random_state=142
    )
    rs_clf_cntrl = clone(rs_clf).fit(X_train[train_is_treatment == 0], y_train[train_is_treatment == 0])
    print(f"Control best params: {rs_clf_cntrl.best_params_}")
    print(f"Control validation score: {rs_clf_cntrl.best_estimator_.score(X_valid[valid_is_treatment == 0], y_valid[valid_is_treatment == 0])}")
    clf_control = rs_clf_cntrl.best_estimator_

    rs_clf_trtmnt = clone(rs_clf).fit(X_train[train_is_treatment == 1], y_train[train_is_treatment == 1])
    print(f"Treatment best params: {rs_clf_trtmnt.best_params_}")
    print(f"Treatment validation score: {rs_clf_trtmnt.best_estimator_.score(X_valid[valid_is_treatment == 1], y_valid[valid_is_treatment == 1])}")
    clf_treatment = rs_clf_trtmnt.best_estimator_

    predict_valid_control = clf_control.predict_proba(X_valid)[:, 1]
    predict_valid_treatment = clf_treatment.predict_proba(X_valid)[:, 1]
    predict_valid_uplift = predict_valid_treatment - predict_valid_control
    valid_uplift_score = uplift_score(predict_valid_uplift, valid_is_treatment, y_valid)
    print(f"Uplift score on validation = {valid_uplift_score}, "
          f"(baseline on validation = 0.05081166028966111, "
          f"difference = {valid_uplift_score - 0.05081166028966111}")

    # JOIN TRAIN AND VALIDATION SETS
    X_control = pd.concat([X_train[train_is_treatment == 0], X_valid[valid_is_treatment == 0]], ignore_index=False)
    X_treatment = pd.concat([X_train[train_is_treatment == 1], X_valid[valid_is_treatment == 1]], ignore_index=False)

    y_control = pd.concat([y_train[train_is_treatment == 0], y_valid[valid_is_treatment == 0]], ignore_index=False)
    y_treatment = pd.concat([y_train[train_is_treatment == 1], y_valid[valid_is_treatment == 1]], ignore_index=False)
    #

    # CROSS VALIDATION SCORE ON WHOLE TRAIN DATASET
    # kfold = StratifiedKFold(n_splits=10, random_state=42)
    #
    # results_control = cross_val_score(clf, X_control, y_control, cv=kfold)
    # print("CV accuracy control: %.2f%% (%.2f%%)" % (results_control.mean() * 100, results_control.std() * 100))
    #
    # results_treatment = cross_val_score(clf, X_treatment, y_treatment, cv=kfold)
    # print("CV accuracy treatment: %.2f%% (%.2f%%)" % (results_treatment.mean() * 100, results_treatment.std() * 100))

    # FITTING ON WHOLE TRAINING SET
    print("fitting classifiers on whole training set")
    clf_control = clf_control.fit(X_control, y_control)
    clf_treatment = clf_treatment.fit(X_treatment, y_treatment)
    #

    # SAVING MODEL AND SUBMISSION
    dt = datetime.now().strftime("%Y-%m-%d_%HH-%MM")
    model_name = dt + "_" + str(clf.__class__).split("'")[1].replace(".", "_")
    f_name = "../../models/two_models/" + model_name

    pickle.dump(
        clf_control, open(f_name + "_control.pkl", "wb")
    )
    pickle.dump(
        clf_treatment, open(f_name + "_treatment.pkl", "wb")
    )

    # with open("../../models/two_models/validation.csv", "a") as f:
    #     f.write(f"{model_name},{valid_uplift_score}\n")

    predict_test_control = clf_control.predict_proba(X_test)[:, 1]
    predict_test_treatment = clf_treatment.predict_proba(X_test)[:, 1]
    predict_test_uplift = predict_test_treatment - predict_test_control
    df_submission = pd.DataFrame({'uplift': predict_test_uplift}, index=X_test.index)
    df_submission.to_csv(f'../../data/submissions/two_models/{model_name}.csv')
