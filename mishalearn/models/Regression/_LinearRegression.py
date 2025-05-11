import matplotlib

from .._Core import BaseLinearRegressor

matplotlib.use("TkAgg")

import numpy as np
import pandas as pd


class OLSRegression:
    def __init__(self):
        self.w: pd.Series = None

    def fit(self, X: pd.DataFrame | pd.Series, y: pd.Series) -> None:
        """
        :param X: Features DataFrame
        :param y: Target Series
        :return: self
        """
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_df.insert(0, 'Intercept', 1.0)
        X_df_t = X_df.T
        xtx_inv = np.linalg.inv(np.dot(X_df_t, X_df))
        xty = np.dot(X_df_t, y)
        w = np.dot(xtx_inv, xty)
        self.w = pd.Series(w, index=X_df.columns)
        return self

    def predict(self, X: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """
        :param X: Features DataFrame
        :return: Target Series
        """
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_df.insert(0, 'Intercept', 1.0)
        predictions = np.dot(X_df, self.w)
        return predictions


class LinearRegression(BaseLinearRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_gradient(self, X, err):
        return 2 / X.shape[0] * np.dot(X.T, err)


class RidgeRegression(BaseLinearRegressor):
    def __init__(self, alpha=0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def _compute_gradient(self, X_batch, err):
        reg_vector = np.concatenate([[0], self._W[1:]])
        return 2 / X_batch.shape[0] * np.dot(X_batch.T, err) + 2 * self.alpha * reg_vector


class LassoRegression(BaseLinearRegressor):
    def __init__(self, alpha=0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def _compute_gradient(self, X_batch, err):
        reg_vector = np.concatenate([[0], np.sign(self._W[1:])])
        return 2 / X_batch.shape[0] * np.dot(X_batch.T, err) + self.alpha * reg_vector


class ElasticNetRegression(BaseLinearRegressor):
    def __init__(self, alpha1=0.01, alpha2=0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def _compute_gradient(self, X_batch, err):
        l1_reg_vector = np.concatenate([[0], np.sign(self._W[1:])])
        l2_reg_vector = np.concatenate([[0], self._W[1:]])
        regularization_term = self.alpha1 * l1_reg_vector + 2 * self.alpha2 * l2_reg_vector
        return 2 / X_batch.shape[0] * np.dot(X_batch.T, err) + regularization_term
