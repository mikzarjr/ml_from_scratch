from sklearn.datasets import (load_diabetes,
                              make_regression)
from sklearn.linear_model import ElasticNet

from mishalearn.metrics import (mean_squared_error,
                                r2_score)
from mishalearn.models import ElasticNet_Regressor
from src.BaseTests import test

my_model = ElasticNet_Regressor()
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

if __name__ == "__main__":
    print(result.to_string(index=False))
