__ALL__ = [
    'OLSRegression',
    'LinearRegression',
    'RidgeRegression',
    'LassoRegression',
    'ElasticNetRegression',
    'KNN',
    'Perceptron_BinaryClassificator',
    'SVM_BinaryClassificator',
    'LogisticRegression_BinaryClassificator',
    'OneVsAllClassifier'
]

from .Classification import (
    KNN,
    Perceptron_BinaryClassificator,
    SVM_BinaryClassificator,
    LogisticRegression_BinaryClassificator,
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
