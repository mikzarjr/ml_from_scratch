from math import sqrt

import numpy as np
import pandas as pd

__ALL__ = [
    'root_mean_squared_error',
    'mean_squared_error',
    'mean_absolute_error'
]


def root_mean_squared_error(y_true: pd.Series, y_pred: pd.Series):
    squared_errors = (y_true - y_pred) ** 2
    return sqrt(np.mean(squared_errors))


def mean_squared_error(y_true: pd.Series, y_pred: pd.Series):
    squared_errors = (y_true - y_pred) ** 2
    return float(np.mean(squared_errors))


def mean_absolute_error(y_true: pd.Series, y_pred: pd.Series):
    abs_errors = np.abs(y_true - y_pred)
    return float(np.mean(abs_errors))
