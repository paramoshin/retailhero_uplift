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
    parser.add_argument("--n_states", type=int, default=1)
    parser.add_argument("--random_states", nargs="+", default=[42])
    args = parser.parse_args()

    assert len(args.random_states) >= args.n_states

    X_train, y_train, train_is_treatment, X_valid, y_valid, valid_is_treatment, X_test = read_train_test()
    X_train, y_train = join_train_validation(X_train, X_valid, y_train, y_valid)
    train_is_treatment = pd.concat([train_is_treatment, valid_is_treatment], ignore_index=False)

    folds_df = pd.DataFrame(
        {f"random_state_{rs}": [0] * X_train.shape[0] for rs in args.random_states}, 
        index=X_train.index
    )

    for rs in args.random_states:
        print(f"Random state {rs}:")
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=int(rs))
        folds = []
        for i, (train_idx, test_idx) in enumerate(skf.split(X_train, y_train)):
            print(len(train_idx), len(test_idx))
            folds_df.loc[folds_df.iloc[test_idx].index, f"random_state_{rs}"] = [i] * len(test_idx)
    folds_df.to_csv("../../data/processed/folds.csv", index=True)
