import matplotlib

from .._Core import BaseLinearRegressor

matplotlib.use("TkAgg")

import numpy as np


class Linear_Regression(BaseLinearRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_gradient(self, X, err):
        return 2 / X.shape[0] * np.dot(X.T, err)


class Ridge_Regression(BaseLinearRegressor):
    def __init__(self, alpha=0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def _compute_gradient(self, X_batch, err):
        reg_vector = np.concatenate([[0], self._W[1:]])
        return 2 / X_batch.shape[0] * np.dot(X_batch.T, err) + 2 * self.alpha * reg_vector


class Lasso_Regression(BaseLinearRegressor):
    def __init__(self, alpha=0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def _compute_gradient(self, X_batch, err):
        reg_vector = np.concatenate([[0], np.sign(self._W[1:])])
        return 2 / X_batch.shape[0] * np.dot(X_batch.T, err) + self.alpha * reg_vector


class ElasticNet_Regression(BaseLinearRegressor):
    def __init__(self, alpha1=0.01, alpha2=0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def _compute_gradient(self, X_batch, err):
        l1_reg_vector = np.concatenate([[0], np.sign(self._W[1:])])
        l2_reg_vector = np.concatenate([[0], self._W[1:]])
        regularization_term = self.alpha1 * l1_reg_vector + 2 * self.alpha2 * l2_reg_vector
        return 2 / X_batch.shape[0] * np.dot(X_batch.T, err) + regularization_term
