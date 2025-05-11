import numpy as np
import pandas as pd


class OLS_Regression:
    def __init__(self):
        super().__init__()
        self.w: pd.Series = None

    def fit(self, X: pd.DataFrame | pd.Series, y: pd.Series) -> None:
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_df.insert(0, 'Intercept', 1.0)
        X_df_t = X_df.T
        xtx_inv = np.linalg.inv(np.dot(X_df_t, X_df))
        xty = np.dot(X_df_t, y)
        w = np.dot(xtx_inv, xty)
        self.w = pd.Series(w, index=X_df.columns)
        return self

    def predict(self, X: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_df.insert(0, 'Intercept', 1.0)
        predictions = np.dot(X_df, self.w)
        return predictions
