"""
This is to be called with the already vectorized lower triangular Laplacians
"""

import numpy as np
import warnings
from sklearn.metrics.cluster import normalized_mutual_info_score

def contingency_table(x, y):
    """
    Computes the contingency table for two binary vectors x and y
    """
    tt = np.sum(np.logical_and(x, y))
    tf = np.sum(np.logical_and(x, np.logical_not(y)))
    ft = np.sum(np.logical_and(np.logical_not(x), y))
    ff = np.sum(np.logical_and(np.logical_not(x), np.logical_not(y)))
    return np.array([[tt, tf], [ft, ff]])

def normalized_mutual_information(x, y):
    """
    Computes the normalized mutual information between two binary vectors x and y
    """
    cont = contingency_table(x, y)
    marg_x = np.sum(cont, axis=1)
    marg_y = np.sum(cont, axis=0)
    n = np.sum(marg_x)
    eps = 1e-8 # a small constant value
    H_x = -(marg_x + eps) @ np.log2((marg_x + eps) / n)
    H_y = -(marg_y + eps) @ np.log2((marg_y + eps) / n)
    p = cont / n
    p[p == 0] = eps # replace zero with a small constant value
    H_xy = -np.sum(p * np.log2(p))
    I_xy = H_x + H_y - H_xy
    NMI = I_xy / ((H_x + H_y) / 2)
    return NMI

def precision_recall(x, y):
    """
    Computes the precision and recall between two binary vectors x and y.
    """
    true_positives = np.sum(np.logical_and(x, y))
    false_positives = np.sum(np.logical_and(np.logical_not(x), y))
    false_negatives = np.sum(np.logical_and(x, np.logical_not(y)))
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f_score = 2/  (1 / precision + 1 / recall)
    return precision, recall, f_score

def threshold_precision_recall(x, y, n_thresholds=30):
    """
    Thresholds the prediction and returns the lists of precision, recall
    and f_score as well as the maximum f_score entry
    """
    scores = []
    thresholds = np.linspace(0,1,n_thresholds)
    thresholds = thresholds[1:-2]

    for t in thresholds:
        aux = x
        aux[aux < t] = 0
        aux[aux > 0] = 1
        scores_threshold = list(precision_recall(aux,y))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nmi_score = normalized_mutual_info_score(aux,y)
        scores_threshold.append(nmi_score)
        scores.append(scores_threshold)

    return scores, max(scores, key=lambda x: x[2])

def mean_squared_error(x, y):
    """
    Computes the mean squared error (MSE) between two vectors x and y.
    """
    mse = np.mean((x - y) ** 2)
    
    return mse