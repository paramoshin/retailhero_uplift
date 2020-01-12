import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV


def optimize(
        X_train,
        y_train,
        n_class=2,
        objective="binary:logistic",
        scoring="roc_auc",
        n_iter=30,
        seed=42,
        params=None,
        cv=None,
        X_val=None,
        y_val=None
    ):
    clf = xgb.XGBClassifier(objective=objective, n_class=n_class)
    if not cv:
        cv = StratifiedKFold(n_splits=5, random_state=seed)
    space = {
        "learning_rate": [0.2, 0.1, 0.05, 0.01],
        "gamma": [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
        "max_depth": [2, 4, 7, 10, 12, 15],
        "colsample_bytree": [0.5, 0.6, 0.8, 1.0],
        "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.5, 1],
        "reg_lambda": [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        "min_child_weight": [1, 3, 5, 7, 12],
        "n_estimators": [100, 250, 500, 1000]
    }
    if params:
        space.update(params)
    rs_clf = RandomizedSearchCV(
        clf, param_distributions=space, scoring=scoring, cv=cv, n_iter=n_iter, verbose=3, random_state=seed, n_jobs=-1
    )
    rs_clf.fit(X_train, y_train)
    print(f"Best parametrs: {rs_clf.best_params_}; best score: {rs_clf.best_score_}")
    if (X_val is not None) and (y_val is not None):
        print(f"Score on valiadtion set: {rs_clf.best_estimator_.score(X_val, y_val)}")
    return rs_clf.best_estimator_, rs_clf.best_params_
