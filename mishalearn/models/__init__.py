__ALL__ = [
]

from .Classification import (
    KNN,
    Perceptron_BinaryClassificator,
    SVM_BinaryClassificator,
    LogisticRegression_BinaryClassificator,

    OvA_MultiClassificator,
    AvA_MultiClassificator
)
from .Regression import (
    OLS_Regression,
    Linear_Regression,
    Ridge_Regression,
    Lasso_Regression,
    ElasticNet_Regression
)
