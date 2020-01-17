import sys
from pathlib import Path
p = str(Path(".").resolve().parent.parent)
sys.path.extend([p])

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

from src.modeling.read_data import *
from src.modeling.utils import uplift_score


if __name__ == "__main__":
    X_train, y_train, train_is_treatment, X_valid, y_valid, valid_is_treatment, X_test = read_train_test()

    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    best_params = {
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
    clf_control = xgb.XGBClassifier(**best_params)
    clf_treatment = xgb.XGBClassifier(**best_params)

    X, y = join_train_validation(X_train, X_valid, y_train, y_valid)
    is_treatment = pd.concat([train_is_treatment, valid_is_treatment], ignore_index=False)

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"FOLD {i + 1}")
        X_train, y_train, X_test, y_test = (
            X.iloc[train_index], y.iloc[train_index], X.iloc[test_index], y.iloc[test_index]
        )

        train_is_treatment = is_treatment.iloc[train_index]
        test_is_treatment = is_treatment.iloc[test_index]

        X_train_control, X_train_treatment, y_train_control, y_train_treatment = \
            split_control_treatment(X.iloc[train_index], y.iloc[train_index], train_is_treatment)
        X_test_control, X_test_treatment, y_test_control, y_test_treatment = \
            split_control_treatment(X.iloc[test_index], y.iloc[test_index], test_is_treatment)

        clf_control.fit(X_train_control, y_train_control)
        print(f"    Control classifier score on train: {clf_control.score(X_train_control, y_train_control)}")
        print(f"    Control classifier score on test: {clf_control.score(X_test, y_test)}\n")

        clf_treatment.fit(X_train_treatment, y_train_treatment)
        print(f"    Treatment classifier score on train: {clf_control.score(X_train_treatment, y_train_treatment)}")
        print(f"    Treatment classifier score on test: {clf_treatment.score(X_test, y_test)}\n")

        uplift_prediction_train = clf_treatment.predict_proba(X_train)[:, 1] - clf_control.predict_proba(X_train)[:, 1]
        print(f"    Predicted uplift score on train: {uplift_score(uplift_prediction_train, train_is_treatment, y_train)}")
        print(f"    True uplift score on train: {uplift_score(y_train, train_is_treatment, y_train)}\n")

        uplift_prediction_test = clf_treatment.predict_proba(X_test)[:, 1] - clf_control.predict_proba(X_test)[:, 1]
        print(f"    Predicted uplift score on test: {uplift_score(uplift_prediction_test, test_is_treatment, y_test)}")
        print(f"    True uplift score on test: {uplift_score(y_test, test_is_treatment, y_test)}\n")
