from datetime import datetime
import pickle
import json

import pandas as pd

from .read_data import *
from .random_search_optimization import optimize

if __name__ == "__main__":
    X_train, y_train, train_is_treatment, X_valid, y_valid, valid_is_treatment, X_test = read_train_test()

    y_train = (((train_is_treatment == 1) & (y_train == 1)) | ((train_is_treatment == 0) & (y_train == 0))).astype(int)
    y_valid = (((valid_is_treatment == 1) & (y_valid == 1)) | ((valid_is_treatment == 0) & (y_valid == 0))).astype(int)

    X_train, y_train = join_train_validation(X_train, X_valid, y_train, y_valid)
    model, best_params = optimize(X_train, y_train)

    dt = datetime.now().strftime("%Y-%m-%d_%HH-%MM")
    model_name = "class_transform_" + dt + "_" + str(model.__class__).split("'")[1].replace(".", "_")
    f_name = "../../models/class_transform/" + model_name

    pickle.dump(model, open(f_name + ".pkl", "wb"))
    with open(f_name + "best_params.json", "w") as f:
        f.write(json.dumps(best_params))

    predict_test_proba = model.predict_proba(X_test)[:, 1]
    predict_test_uplift = 2 * predict_test_proba - 1
    df_submission = pd.DataFrame({'uplift': predict_test_uplift}, index=X_test.index)
    df_submission.to_csv(f'../../data/submissions/class_transform/{model_name}.csv')
