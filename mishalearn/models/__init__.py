__ALL__ = [
    'OLSRegression',
    'LinearRegression',
    'RidgeRegression',
    'LassoRegression',
    'ElasticNetRegression',
    'KNN',
    'LinearClassification',
    'SVM',
    'LogisticRegression',
    'OneVsAllClassifier'
]

from .Classification import (
    KNN,
    LinearClassification,
    SVM,
    LogisticRegression,
    OneVsAllClassifier,
    AllVsAllClassifier
)
from .Regression import (
    OLSRegression,
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    ElasticNetRegression
)
