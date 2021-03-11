# Created by Hansi at 7/3/2020
from sklearn.metrics import f1_score, recall_score, precision_score

from examples.common.converter import decode


def f1(y_true, y_pred):
    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))
    return f1_score(y_true, y_pred)


def recall(y_true, y_pred):
    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))
    return recall_score(y_true, y_pred)


def precision(y_true, y_pred):
    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))
    return precision_score(y_true, y_pred)


def confusion_matrix(y_true, y_pred):
    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))
    return confusion_matrix(y_pred, y_true).ravel()
