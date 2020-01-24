from datetime import datetime
import pickle
import json
import sys
from pathlib import Path
p = str(Path(".").resolve().parent.parent)
sys.path.extend([p])

import pandas as pd
import xgboost as xgb

from src.modeling.utils import *
from src.modeling.random_search_optimization import optimize


if __name__ == "__main__":
    X_train, y_train, train_is_treatment, X_valid, y_valid, valid_is_treatment, X_test = read_train_test()

    X_train_control, X_train_treatment, y_train_control, y_train_treatment = split_control_treatment(
        X_train, y_train, train_is_treatment
    )
    X_valid_control, X_valid_treatment, y_valid_control, y_valid_treatment = split_control_treatment(
        X_valid, y_valid, valid_is_treatment
    )

    clf_control, best_params_control = optimize(
        X_train_control, y_train_control, X_val=X_valid_control, y_val=y_valid_control
    )
    clf_treatment, best_params_treatment = optimize(
        X_train_treatment, y_train_treatment, X_val=X_valid_treatment, y_val=y_valid_treatment
    )
    valid_uplift = clf_treatment.predict_proba(X_valid)[:, 1] - clf_control.predict_proba(X_valid)[:, 1]
    valid_uplift_score = uplift_score(valid_uplift, valid_is_treatment, y_valid)
    print(f"Uplit score on validation: {valid_uplift_score}")

    # JOIN TRAIN AND VALIDATION SETS
    X_control, y_control = join_train_validation(X_train_control, X_valid_control, y_train_control, y_valid_control)
    X_treatment, y_treatment = join_train_validation(
        X_train_treatment, X_valid_treatment, y_train_treatment, y_valid_treatment
    )
    clf_control = xgb.XGBClassifier(objective="binary:logistic", **best_params_control).fit(X_control, y_control)
    clf_treatment = xgb.XGBClassifier(objective="binary:logistic", **best_params_treatment).fit(X_treatment, y_treatment)

    # SAVING MODEL AND SUBMISSION
    dt = datetime.now().strftime("%Y-%m-%d_%HH-%MM")
    model_name = "two_models_" + dt + "_" + str(clf_control.__class__).split("'")[1].replace(".", "_")
    f_name = "../../models/two_models/" + model_name

    with open(f_name + "best_params_control.json", "w") as f:
        f.write(json.dumps(best_params_control))
    with open(f_name + "best_params_treatment.json", "w") as f:
        f.write(json.dumps(best_params_treatment))

    pickle.dump(
        clf_control, open(f_name + "_control.pkl", "wb")
    )
    pickle.dump(
        clf_treatment, open(f_name + "_treatment.pkl", "wb")
    )

    predict_test_control = clf_control.predict_proba(X_test)[:, 1]
    predict_test_treatment = clf_treatment.predict_proba(X_test)[:, 1]
    predict_test_uplift = predict_test_treatment - predict_test_control
    df_submission = pd.DataFrame({'uplift': predict_test_uplift}, index=X_test.index)
    df_submission.to_csv(f'../../data/submissions/two_models/{model_name}.csv')
