__all__ = ["uplift_score"]

import numpy as np


def uplift_score(prediction, treatment, target, rate=0.3):
    """
    Подсчет Uplift Score
    """
    order = np.argsort(-prediction)
    treatment_n = int((treatment == 1).sum() * rate)
    print(f"    number of treatment users: {treatment_n}")
    treatment_p = target[order][treatment[order] == 1][:treatment_n].mean()
    print(f"    treatment p: {treatment_p}")
    control_n = int((treatment == 0).sum() * rate)
    print(f"    number of control users: {treatment_n}")
    control_p = target[order][treatment[order] == 0][:control_n].mean()
    print(f"    control p: {treatment_p}")
    score = treatment_p - control_p
    return score
