__ALL__ = [
    'KNN',
    'LinearClassification',
    'SVM',
    'LogisticRegression',
    'OneVsAllClassifier'
]

from ._Binary import (
    LinearClassification,
    SVM,
    LogisticRegression
)
from ._KNN import KNN
from ._Multi import (
    OneVsAllClassifier,
    AllVsAllClassifier
)
