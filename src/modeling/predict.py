from datetime import datetime
import sys
from pathlib import Path
p = str(Path(".").resolve().parent.parent)
sys.path.extend([p])

import pandas as pd
import joblib

from src.modeling.read_data import *

if __name__ == "__main__":
    X_train, y_train, train_is_treatment, X_valid, y_valid, valid_is_treatment, X_test = read_train_test()
    X_train, y_train = join_train_validation(X_train, X_valid, y_train, y_valid)
    train_is_treatment = pd.concat([train_is_treatment, valid_is_treatment], ignore_index=False)
    folds = pd.read_csv("../../data/processed/folds.csv", index_col="client_id")
    dt = datetime.now().strftime("%Y-%m-%d-%H-%M")

    for i in range(5):
        print(f"Fold {i + 1}")

        folders = list(Path(f"../../models/two_models/folds/").glob("*"))
        folder = sorted(folders)[-1]
        control_path = list(folder.glob(f"*_{i}_control.pkl"))[0]
        treatment_path = list(folder.glob(f"*_{i}_treatment.pkl"))[0]
        clf_control = joblib.load(control_path)
        clf_treatment = joblib.load(treatment_path)

        treatment_proba = clf_treatment.predict_proba(X_test)[:, 1]
        control_proba = clf_control.predict_proba(X_test)[:, 1]
        uplift_prediction = treatment_proba - control_proba
        if i == 0:
            preds = uplift_prediction
        else:
            preds += uplift_prediction

    preds /= 5
    df_submission = pd.DataFrame({'uplift': preds}, index=X_test.index)
    submission_folder = Path(f"../../data/submissions/two_models/folds")
    submission_folder.mkdir(exist_ok=True)
    df_submission.to_csv(f'{submission_folder}/submission_{dt}.csv')
