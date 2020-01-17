from datetime import datetime
import sys
from pathlib import Path
p = str(Path(".").resolve().parent.parent)
sys.path.extend([p])

import pandas as pd
# from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgbm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from src.modeling.read_data import *
from src.modeling.utils import uplift_score


if __name__ == "__main__":
    X_train, y_train, train_is_treatment, X_valid, y_valid, valid_is_treatment, X_test = read_train_test()

    X_train = X_train.fillna(999)
    X_valid = X_valid.fillna(999)
    X_test = X_test.fillna(999)

    X_train_control, X_train_treatment, y_train_control, y_train_treatment = split_control_treatment(
        X_train, y_train, train_is_treatment
    )
    X_valid_control, X_valid_treatment, y_valid_control, y_valid_treatment = split_control_treatment(
        X_valid, y_valid, valid_is_treatment
    )
    params = {'learning_rate': 0.03, 'max_depth': 4, 'num_leaves': 20,
              'min_data_in_leaf': 3, 'application': 'binary',
              'subsample': 0.8, 'colsample_bytree': 0.8,
              'reg_alpha': 0.01, 'data_random_seed': 42, 'metric': 'binary_logloss',
              'max_bin': 416, 'bagging_freq': 3, 'reg_lambda': 0.01
              }
    matrix = lgbm.Dataset(X_train_control, label=y_train_control)
    cv_result = lgbm.cv(params, matrix, num_boost_round=5000, nfold=5, stratified=True,
                        early_stopping_rounds=50, seed=42, verbose_eval=50)

    clf_control = lgbm.LGBMClassifier(n_estimators=len(cv_result['binary_logloss-mean']), **params)
    clf_control.fit(X_train_control, y_train_control)
    print(f"Control score on validation: {clf_control.score(X_valid_control, y_valid_control)}")
    clf_treatment = lgbm.LGBMClassifier(n_estimators=len(cv_result['binary_logloss-mean']), **params)
    clf_treatment.fit(X_train_treatment, y_train_treatment)
    print(f"Treatment score on validation: {clf_treatment.score(X_valid_treatment, y_valid_treatment)}")

    valid_uplift = clf_treatment.predict_proba(X_valid)[:, 1] - clf_control.predict_proba(X_valid)[:, 1]
    valid_uplift_score = uplift_score(valid_uplift, valid_is_treatment, y_valid)
    print(f"Uplit score on validation: {valid_uplift_score}")

    X_control, y_control = join_train_validation(X_train_control, X_valid_control, y_train_control, y_valid_control)
    X_treatment, y_treatment = join_train_validation(
        X_train_treatment, X_valid_treatment, y_train_treatment, y_valid_treatment
    )
    clf_control = lgbm.LGBMClassifier(n_estimators=len(cv_result['binary_logloss-mean']), **params)
    clf_control.fit(X_control, y_control)
    clf_treatment = lgbm.LGBMClassifier(n_estimators=len(cv_result['binary_logloss-mean']), **params)
    clf_treatment.fit(X_treatment, y_treatment)

    dt = datetime.now().strftime("%Y-%m-%d_%HH-%MM")
    model_name = "two_models_" + dt + "_" + str(clf_control.__class__).split("'")[1].replace(".", "_")
    f_name = "../../models/two_models/" + model_name

    feature_imp = pd.DataFrame(
        sorted(zip(clf_treatment.feature_importances_, X_treatment.columns)), columns=['Value', 'Feature']
    )
    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'../../data/lgbm_importances-01.png')

    predict_test_control = clf_control.predict_proba(X_test)[:, 1]
    predict_test_treatment = clf_treatment.predict_proba(X_test)[:, 1]
    predict_test_uplift = predict_test_treatment - predict_test_control
    df_submission = pd.DataFrame({'uplift': predict_test_uplift}, index=X_test.index)
    df_submission.to_csv(f'../../data/submissions/two_models/{model_name}.csv')
