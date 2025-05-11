import pandas as pd

__ALL__ = [
    'accuracy',
    'precision',
    'recall',
    'f1_score'
]


def get_tp(y_true: pd.Series, y_pred: pd.Series, positive):
    tp_mask = (y_true == positive) & (y_pred == positive)
    fp_mask = (y_true != positive) & (y_pred == positive)
    tn_mask = (y_true != positive) & (y_pred != positive)
    fn_mask = (y_true == positive) & (y_pred != positive)

    TP = tp_mask.sum()
    FP = fp_mask.sum()
    TN = tn_mask.sum()
    FN = fn_mask.sum()

    return TP, FP, TN, FN


def accuracy(y_true: pd.Series, y_pred: pd.Series):
    return sum(y_true == y_pred) / len(y_true)


def precision(y_true: pd.Series, y_pred: pd.Series):
    labels = sorted(y_true.unique())
    precisions = []
    for positive in labels:
        TP_c, FP_c, _, _ = get_tp(y_true, y_pred, positive)
        precisions.append(TP_c / (TP_c + FP_c))

    return sum(precisions) / len(labels)


def recall(y_true: pd.Series, y_pred: pd.Series):
    labels = sorted(y_true.unique())
    recalls = []
    for positive in labels:
        TP_c, _, _, FN_c = get_tp(y_true, y_pred, positive)
        recalls.append(TP_c / (TP_c + FN_c))

    return sum(recalls) / len(labels)


def f1_score(y_true: pd.Series, y_pred: pd.Series):
    labels = sorted(y_true.unique())
    K = len(labels)
    F1 = 0
    for positive in labels:
        TP_c, FP_c, TN_c, FN_c = get_tp(y_true, y_pred, positive)
        precision_c = TP_c / (TP_c + FP_c) if TP_c + FP_c > 0 else 0.0
        recall_c = TP_c / (TP_c + FN_c) if TP_c + FN_c > 0 else 0.0
        F1_c = 2 * precision_c * recall_c / (precision_c + recall_c) if (precision_c + recall_c) > 0 else 0.0
        F1 += F1_c
    return F1 / K
