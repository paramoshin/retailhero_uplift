__all__ = ["models"]

from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

models = {
    "xgb": XGBClassifier(),
    "randomforest": ensemble.RandomForestClassifier(n_estimators=100),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=100),
    "lightgbm": LGBMClassifier(),
    "logreg": LogisticRegression(),
    "knn": KNeighborsClassifier(),
    "gradientboosting": ensemble.GradientBoostingClassifier(n_estimators=100),
}