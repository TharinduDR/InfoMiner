# Created by Hansi at 7/3/2020
from sklearn.metrics import f1_score, recall_score, precision_score

from examples.common.converter import decode

labels = ['NoADE', 'ADE']
pos_label = 'ADE'


def f1(y_true, y_pred):
    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))
    y_true = decode(y_true)
    y_pred = decode(y_pred)
    return f1_score(y_true, y_pred, labels=labels, pos_label=pos_label)


def recall(y_true, y_pred):
    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))
    y_true = decode(y_true)
    y_pred = decode(y_pred)
    return recall_score(y_true, y_pred, labels=labels, pos_label=pos_label)


def precision(y_true, y_pred):
    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))
    y_pred = decode(y_pred)
    return precision_score(y_true, y_pred, labels=labels, pos_label=pos_label)
