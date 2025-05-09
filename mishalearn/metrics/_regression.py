from math import sqrt

import numpy as np
import pandas as pd

__ALL__ = [
    'root_mean_squared_error',
    'mean_squared_error',
    'mean_absolute_error',
    'r2_score',
    'mean_absolute_percentage_error'
]


def root_mean_squared_error(y_true: pd.Series, y_pred: pd.Series):
    return sqrt(np.mean((y_true - y_pred) ** 2))


def mean_squared_error(y_true: pd.Series, y_pred: pd.Series, squared: bool = True):
    mse = np.mean((y_true - y_pred) ** 2)
    if squared:
        return mse
    return sqrt(mse)


def mean_absolute_error(y_true: pd.Series, y_pred: pd.Series):
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true: pd.Series, y_pred: pd.Series):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)


def mean_absolute_percentage_error(y_true: pd.Series, y_pred: pd.Series):
    return np.mean(np.abs((y_true - y_pred) / y_true))
