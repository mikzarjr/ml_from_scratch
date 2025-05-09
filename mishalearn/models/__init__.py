__ALL__ = [
    'OLSRegression',
    'LinearRegression',
    'RidgeRegression',
    'LassoRegression',
    'ElasticNetRegression',
    'KNN',
    'LinearClassification',
    'SVM',
    'LogisticRegression'
]

from .Classification import (
    KNN,
    LinearClassification,
    SVM,
    LogisticRegression
)
from .Regression import (
    OLSRegression,
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    ElasticNetRegression
)
