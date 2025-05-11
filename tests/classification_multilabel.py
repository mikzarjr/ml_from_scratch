from sklearn.datasets import (load_digits,
                              make_classification)
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

from src.BaseTests import test
from mishalearn.metrics import (accuracy,
                                f1_score)
from mishalearn.models import (AvA_MultiClassificator,
                               SVM_BinaryClassificator)

my_model = AvA_MultiClassificator(base_clf_cls=SVM_BinaryClassificator)
sk_model = OneVsOneClassifier(estimator=LinearSVC())

datasets = [
    ("Digits", load_digits),
    ("SyntheticBin", make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=7,
        n_classes=10,
        random_state=0
    ))
]

metrics = [
    accuracy,
    f1_score
]

result = test(my_model, sk_model, datasets, metrics)
print(result.to_string(index=False))
