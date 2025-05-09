__ALL__ = [
    'KNN',
    'LinearClassification',
    'SVM',
    'LogisticRegression'
]

from ._Binary import (
    LinearClassification,
    SVM,
    LogisticRegression
)
from ._Multi import KNN
