from sklearn.datasets import (load_diabetes,
                              make_regression)
from sklearn.linear_model import ElasticNet

from base import test
from mishalearn.metrics import (mean_squared_error,
                                r2_score)
from mishalearn.models import ElasticNet_Regression

my_model = ElasticNet_Regression()
sk_model = ElasticNet()

datasets = [
    ("Diabetes", load_diabetes),
    ("SyntheticReg", make_regression(
        n_samples=10000,
        n_features=20,
        random_state=0
    ))
]
metrics = [
    mean_squared_error,
    r2_score
]
result = test(my_model, sk_model, datasets, metrics)
print(result.to_string(index=False))
