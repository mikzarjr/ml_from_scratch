__ALL__ = [
    'root_mean_squared_error',
    'mean_squared_error',
    'mean_absolute_error',
    'r2_score',
    'mean_absolute_percentage_error'

    'accuracy'
]

from ._classification import (
    accuracy,
    precision,
    recall,
    f1_score
)
from ._regression import (
    root_mean_squared_error,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
