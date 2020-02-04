import argparse
from datetime import datetime
import sys
from pathlib import Path
p = str(Path(".").resolve().parent.parent)
sys.path.extend([p])

import pandas as pd
import joblib
from mlflow import log_param, log_artifact

from src.modeling.utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=str, default=None)
    parser.add_argument("--to_average", type=str, default="uplift")
    args = parser.parse_args()

    log_param("model-dt", args.dt)

    dt = datetime.now().strftime("%Y-%m-%d-%H-%M")

    X_train, y_train, train_is_treatment, X_valid, y_valid, valid_is_treatment, X_test = read_train_test()
    X_train, y_train = join_train_validation(X_train, X_valid, y_train, y_valid)
    train_is_treatment = pd.concat([train_is_treatment, valid_is_treatment], ignore_index=False)
    folds = pd.read_csv("../../data/processed/folds.csv", index_col="client_id")

    for i in range(5):
        print(f"Fold {i + 1}")
        if not args.dt:
            folders = list(Path(f"../../models/two_models/folds/").glob("*"))
            folder = sorted(folders)[-1]
        else:
            folder = Path(f"../../models/two_models/folds/{dt}")
        control_path = list(folder.glob(f"*_{i}_control.pkl"))[0]
        treatment_path = list(folder.glob(f"*_{i}_treatment.pkl"))[0]
        clf_control = joblib.load(control_path)
        clf_treatment = joblib.load(treatment_path)

        treatment_proba = clf_treatment.predict_proba(X_test)[:, 1]
        control_proba = clf_control.predict_proba(X_test)[:, 1]
        if args.to_average == "uplift":
            uplift_prediction = treatment_proba - control_proba
            if i == 0:
                preds = uplift_prediction
            else:
                preds += uplift_prediction
        else:
            if i == 0:
                treatment_preds = treatment_proba
                control_preds = control_proba
            else:
                treatment_preds += treatment_proba
                control_preds += control_proba
                
    if args.to_average == "uplift":
        preds /= 5
        df_submission = pd.DataFrame({'uplift': preds}, index=X_test.index)
    else:
        treatment_preds /= 5
        control_preds /= 5
        uplift_prediction = treatment_preds - control_preds
        df_submission = pd.DataFrame({'uplift': uplift_prediction}, index=X_test.index)

    submission_folder = Path(f"../../data/submissions/folds")
    submission_folder.mkdir(parents=True, exist_ok=True)
    df_submission.to_csv(f'{submission_folder}/submission_{dt}.csv')
    log_artifact(f'{submission_folder}/submission_{dt}.csv', "submission")

