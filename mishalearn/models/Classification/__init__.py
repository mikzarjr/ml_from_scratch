__ALL__ = [

]

from ._Binary import (
    Perceptron_BinaryClassificator,
    SVM_BinaryClassificator,
    LogisticRegression_BinaryClassificator
)
from ._KNN import KNN
from ._Multi import (
    OvA_MultiClassificator,
    AvA_MultiClassificator
)
