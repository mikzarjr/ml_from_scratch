import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mishalearn.metrics import root_mean_squared_error, r2_score
from mishalearn.models import (
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    ElasticNetRegression,
    OLSRegression
)


def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"{name:<20} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
    return rmse, r2


def main():
    # Загружаем датасет
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target

    # Нормализуем признаки
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Разбиваем на трейн/тест
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        "OLSRegression": OLSRegression(),
        "LinearRegression": LinearRegression(lr=0.01, max_steps=3000, show_graph=False, verbose=False),
        "RidgeRegression": RidgeRegression(alpha=0.1, lr=0.01, max_steps=3000, show_graph=False, verbose=False),
        "LassoRegression": LassoRegression(alpha=0.1, lr=0.01, max_steps=3000, show_graph=False, verbose=False),
        "ElasticNetRegression": ElasticNetRegression(alpha1=0.05, alpha2=0.05, lr=0.01, max_steps=3000,
                                                     show_graph=False, verbose=False),
    }

    print(f"{'Model':<20} | {'RMSE':<10} | {'R²':<10}")
    print("-" * 50)

    results = {}

    for name, model in models.items():
        rmse, r2 = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results[name] = (rmse, r2)

    return results


if __name__ == "__main__":
    main()
