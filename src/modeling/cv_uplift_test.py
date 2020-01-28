import sys
from pathlib import Path
p = str(Path(".").resolve().parent.parent)
sys.path.extend([p])

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

from src.modeling.utils import *


if __name__ == "__main__":
    X_train, y_train, train_is_treatment, X_valid, y_valid, valid_is_treatment, X_test = read_train_test()

    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    
    # best_params = {
    #     "subsample": 0.7,
    #     "reg_lambda": 1.0,
    #     "reg_alpha": 0,
    #     "n_estimators": 1000,
    #     "min_child_weight": 7,
    #     "max_depth": 4,
    #     "learning_rate": 0.01,
    #     "gamma": 0.3,
    #     "colsample_bytree": 0.5
    # }
    clf_control = xgb.XGBClassifier(random_state=42, verbosity=1)
    clf_treatment = xgb.XGBClassifier(random_state=42, verbosity=1)

    X, y = join_train_validation(X_train, X_valid, y_train, y_valid)
    is_treatment = pd.concat([train_is_treatment, valid_is_treatment], ignore_index=False)

    control_accs = []
    control_aucs = []
    treatment_accs = []
    treatment_aucs = []
    uplift_scores = []

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
        control_acc = accuracy_score(y_test, clf_control.predict(X_test))
        control_auc = roc_auc_score(y_test, clf_control.predict_proba(X_test)[:, 1])
        control_accs.append(control_acc)
        control_aucs.append(control_auc)

        clf_treatment.fit(X_train_treatment, y_train_treatment)
        treatment_acc = accuracy_score(y_test, clf_treatment.predict(X_test))
        treatment_auc = roc_auc_score(y_test, clf_treatment.predict_proba(X_test)[:, 1])
        treatment_accs.append(treatment_acc)
        treatment_aucs.append(treatment_auc)

        uplift_prediction_test = clf_treatment.predict_proba(X_test)[:, 1] - clf_control.predict_proba(X_test)[:, 1]    
        uplift_scores.append(uplift_score(uplift_prediction_test, test_is_treatment, y_test))
    
    print(f"Control clf accuracy on folds: {control_accs}")
    print(f"Control clf roc_auc on folds: {control_aucs}")

    print(f"Control clf average accuracy: {np.mean(control_accs)}")
    print(f"Control clf average roc_auc: {np.mean(control_auc)}")

    print(f"Treatment clf accuracy on folds: {treatment_accs}")
    print(f"Treatment clf accuracy on folds: {treatment_aucs}")

    print(f"Treatment clf average accuracy: {np.mean(treatment_accs)}")
    print(f"Treatment clf average roc_auc: {np.mean(treatment_aucs)}")

    print(f"Uplift score on folds: {uplift_scores}")
    print(f"Average uplift score: {np.mean(uplift_scores)}")
