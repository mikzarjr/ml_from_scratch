__ALL__ = [
    'OLSRegression',
    'LinearRegression',
    'RidgeRegression',
    'LassoRegression',
    'ElasticNetRegression',
    'KNN',
    'LinearClassification'
]

from .Classification import (
    KNN,
    LinearClassification
)
from .Regression import (
    OLSRegression,
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    ElasticNetRegression
)
