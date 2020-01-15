from datetime import datetime
import pickle
import json
import sys
from pathlib import Path
p = str(Path(".").resolve().parent.parent)
sys.path.extend([p])

import pandas as pd
from sklearn.base import clone

from src.modeling.read_data import *
from src.modeling.random_search_optimization import optimize
from src.modeling.utils import uplift_score

if __name__ == "__main__":
    X_train, y_train, train_is_treatment, X_valid, y_valid, valid_is_treatment, X_test = read_train_test()

    X_train["new_target"] = 0
    X_train.loc[(train_is_treatment == 0) & (y_train == 1), "new_target"] = 1
    X_train.loc[(train_is_treatment == 1) & (y_train == 0), "new_target"] = 2
    X_train.loc[(train_is_treatment == 1) & (y_train == 1), "new_target"] = 3
    new_y_train = X_train["new_target"]
    X_train.drop("new_target", axis=1, inplace=True)

    X_valid["new_target"] = 0
    X_valid.loc[(valid_is_treatment == 0) & (y_valid == 1), "new_target"] = 1
    X_valid.loc[(valid_is_treatment == 1) & (y_valid == 0), "new_target"] = 2
    X_valid.loc[(valid_is_treatment == 1) & (y_valid == 1), "new_target"] = 3
    new_y_valid = X_valid["new_target"]
    X_valid.drop("new_target", axis=1, inplace=True)

    model, best_params = optimize(
        X_train, y_train, num_class=4, objective="multi:softprob", scoring="accuracy", X_val=X_valid, y_val=y_valid
    )

    X_train, y_train = join_train_validation(X_train, X_valid, y_train, y_valid)
    model = clone(model).fit(X_train, y_train)

    dt = datetime.now().strftime("%Y-%m-%d_%HH-%MM")
    model_name = "multiclass" + dt + "_" + str(model.__class__).split("'")[1].replace(".", "_")
    f_name = "../../models/multiclass/" + model_name

    pickle.dump(model, open(f_name + ".pkl", "wb"))
    with open(f_name + "best_params.json", "w") as f:
        f.write(json.dumps(best_params))

    class_probs = model.predict_proba(X_test)
    test_uplift = class_probs[:, 0] + class_probs[:, 3] - class_probs[:, 1] - class_probs[:, 2]
    df_submission = pd.DataFrame({'uplift': test_uplift}, index=X_test.index)
    df_submission.to_csv(f'../../data/submissions/multiclass/{model_name}.csv')
