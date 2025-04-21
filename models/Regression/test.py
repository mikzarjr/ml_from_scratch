import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from models.Regression.LinearRegression import LIN_Regression, OLS_Regression

housing = fetch_california_housing(as_frame=True).frame.dropna()
X = housing.drop(columns='MedHouseVal')
y = housing['MedHouseVal']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

kf = KFold(n_splits=5, shuffle=True, random_state=0)
results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
    X_tr, X_te = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    model = OLS_Regression()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    rmse = mean_squared_error(y_te, preds)

    r2 = r2_score(y_te, preds)
    results.append({'Fold': fold, 'RMSE': rmse, 'R2': r2})

df_res = pd.DataFrame(results)

print(f"Среднее RMSE: {df_res['RMSE'].mean():.4f}")
print(f"Среднее R²:   {df_res['R2'].mean():.4f}")

results = []
for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
    X_tr, X_te = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    model = LinearRegression()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    rmse = mean_squared_error(y_te, preds)

    r2 = r2_score(y_te, preds)
    results.append({'Fold': fold, 'RMSE': rmse, 'R2': r2})

df_res = pd.DataFrame(results)

print(f"Среднее RMSE: {df_res['RMSE'].mean():.4f}")
print(f"Среднее R²:   {df_res['R2'].mean():.4f}")
