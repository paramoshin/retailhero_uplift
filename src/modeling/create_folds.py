import argparse
import sys
from pathlib import Path
p = str(Path(".").resolve().parent.parent)
sys.path.extend([p])

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.modeling.utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    X_train, y_train, train_is_treatment, X_valid, y_valid, valid_is_treatment, X_test = read_train_test()
    X_train, y_train = join_train_validation(X_train, X_valid, y_train, y_valid)
    train_is_treatment = pd.concat([train_is_treatment, valid_is_treatment], ignore_index=False)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    folds = []
    for i, (train_idx, test_idx) in enumerate(skf.split(X_train, y_train)):
        print(len(train_idx), len(test_idx))
        folds.append(pd.DataFrame({"client_id": X_train.iloc[test_idx].index, "fold": [i] * len(test_idx)}))
    folds_df = pd.concat(folds, ignore_index=False)
    folds_df.to_csv("../../data/processed/folds.csv", index=False)
