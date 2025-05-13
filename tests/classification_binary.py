from sklearn.datasets import (load_breast_cancer,
                              make_classification)
from sklearn.linear_model import Perceptron

from mishalearn.metrics import (accuracy,
                                f1_score)
from mishalearn.models import (LogisticRegression_BinaryClassificator)
from src.BaseTests import test

my_model = LogisticRegression_BinaryClassificator()
sk_model = Perceptron(random_state=0)

datasets = [
    ("BreastCancer", load_breast_cancer),
    ("SyntheticBin", make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=17,
        n_classes=2,
        random_state=0
    ))
]

metrics = [
    accuracy,
    f1_score
]

result = test(my_model, sk_model, datasets, metrics)

if __name__ == "__main__":
    print(result.to_string(index=False))
