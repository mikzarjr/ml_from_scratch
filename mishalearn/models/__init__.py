__ALL__ = [
]

from .Classification import (
    KNN,
    Perceptron_BinaryClassificator,
    SVM_BinaryClassificator,
    LogisticRegression_BinaryClassificator,

    OvA_MultiClassificator,
    AvA_MultiClassificator,
    LogisticRegression_MultiClassificator
)
from .Regression import (
    OLS_Regression,
    Linear_Regressor,
    Ridge_Regressor,
    Lasso_Regressor,
    ElasticNet_Regressor
)
