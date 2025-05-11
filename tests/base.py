from typing import Callable

import pandas as pd

from mishalearn.preprocesing import train_test_split


def test(my_model, sk_model, datasets, metrics):
    results = []

    for name, loader in datasets:
        if isinstance(loader, Callable):
            data = loader()
        else:
            data = loader

        if isinstance(data, tuple) and len(data) == 2:
            X, y = data
        else:
            X, y = data.data, data.target

        X = pd.DataFrame(X)
        y = pd.Series(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=0
        )

        my_model.fit(X_train, y_train)
        y_pred_custom = my_model.predict(X_test)

        sk_model.fit(X_train, y_train)
        y_pred_sk = sk_model.predict(X_test)

        results.append({
            f"Dataset": name,
            f"Custom {metrics[0].__name__}": round(metrics[0](y_test, y_pred_custom), 4),
            f"Sklearn {metrics[0].__name__}": round(metrics[0](y_test, y_pred_sk), 4),
            f"Custom {metrics[1].__name__}": round(metrics[1](y_test, y_pred_custom), 4),
            f"Sklearn {metrics[1].__name__}": round(metrics[1](y_test, y_pred_sk), 4),
        })

    df_reg = pd.DataFrame(results)
    return df_reg
