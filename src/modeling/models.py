__all__ = ["models"]

from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

models = {
    "xgb": XGBClassifier(objective="binary:logistic", n_estimators=200, random_state=42, n_jobs=-1),
    "randomforest": ensemble.RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "lightgbm": LGBMClassifier(),
    "logreg": LogisticRegression(),
    "knn": KNeighborsClassifier(),
    "gradientboosting": ensemble.GradientBoostingClassifier(n_estimators=200, random_state=42),
}