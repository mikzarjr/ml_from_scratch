from abc import ABC
from typing import Optional

import numpy as np
import pandas as pd

from mishalearn.models.Classification.Base import BaseClassifier


class BaseLinearClassifier(BaseClassifier, ABC):
    def __init__(
            self,
            max_iter: int,
            lr: float,
            l1_alpha: Optional[float],
            l2_alpha: Optional[float],
            stochastic: bool,
            batch_size: Optional[float]
    ):
        super().__init__()
        self._max_iter = max_iter
        self._lr = lr
        self._l1_alpha = l1_alpha
        self._l2_alpha = l2_alpha
        self._stochastic = stochastic
        self._batch_size = batch_size

    def _fit(self, X: pd.DataFrame | pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> None:
        n = X.shape[0]
        if self._stochastic:
            if self._batch_size is None:
                bs = 1
            else:
                bs = int(self._batch_size * n) if 0 < self._batch_size < 1 else int(self._batch_size)
            bs = max(1, min(bs, n))
        else:
            bs = n

        for _ in range(self._max_iter):
            if self._stochastic:
                idxs = np.random.permutation(n)
            else:
                idxs = np.arange(n)

            for start in range(0, n, bs):
                batch_idxs = idxs[start:start + bs]
                self._regularization()
                self._calculate_gradient(X, y, batch_idxs)

    def _regularization(self):
        if self._l2_alpha is not None:
            self._W[1:] *= (1 - self._lr * self._l2_alpha)
        if self._l1_alpha is not None:
            self._W[1:] -= self._lr * self._l1_alpha * np.sign(self._W[1:])

    @staticmethod
    def _check_classes(y_unique):
        if len(y_unique) != 2:
            raise ValueError("Number of target classes must be 2 for binary classification")
        if y_unique != {0, 1}:
            raise ValueError("Target classes must be formatted as 0 and 1")
        return


class LinearClassification(BaseLinearClassifier):
    def __init__(
            self,
            max_iter: int = 1000,
            lr: float = 0.001,
            l1_alpha: Optional[float] = None,
            l2_alpha: Optional[float] = None,
            stochastic: bool = False,
            batch_size: Optional[float] = None
    ):
        super().__init__(max_iter, lr, l1_alpha, l2_alpha, stochastic, batch_size)

    def _calculate_gradient(self, X_mat, y_vec, batch_idxs):
        for i in batch_idxs:
            x_i, y_i = X_mat[i], y_vec[i]
            if y_i * np.dot(self._W, x_i) <= 0:
                self._W += self._lr * y_i * x_i

    def _calculate_preds(self, X_mat):
        preds = np.sign(np.dot(X_mat, self._W))
        return preds

    def _prepare_classes_forward(self, y: np.ndarray) -> np.ndarray:
        forward_map = {0: -1, 1: +1}
        return np.array([forward_map[label] for label in y])

    def _prepare_classes_backward(self, preds_vec: np.ndarray) -> np.ndarray:
        forward_map = {-1: 0, +1: 1}
        return np.array([forward_map[label] for label in preds_vec])


class SVM(BaseLinearClassifier):
    def __init__(
            self,
            max_iter: int = 1000,
            lr: float = 0.001,
            l1_alpha: Optional[float] = None,
            l2_alpha: Optional[float] = None,
            stochastic: bool = False,
            batch_size: Optional[float] = None
    ):
        super().__init__(max_iter, lr, l1_alpha, l2_alpha, stochastic, batch_size)

    def _calculate_gradient(self, X_mat, y_vec, batch_idxs):
        for i in batch_idxs:
            x_i, y_i = X_mat[i], y_vec[i]
            if 1 - y_i * np.dot(self._W, x_i) > 0:
                self._W += self._lr * y_i * x_i

    def _calculate_preds(self, X_mat):
        preds = np.sign(np.dot(X_mat, self._W))
        return preds

    def _prepare_classes_forward(self, y: np.ndarray) -> np.ndarray:
        forward_map = {0: -1, 1: +1}
        return np.array([forward_map[label] for label in y])

    def _prepare_classes_backward(self, preds_vec: np.ndarray) -> np.ndarray:
        forward_map = {-1: 0, +1: 1}
        return np.array([forward_map[label] for label in preds_vec])


class LogisticRegression(BaseLinearClassifier):
    def __init__(
            self,
            max_iter: int = 1000,
            lr: float = 0.001,
            l1_alpha: Optional[float] = None,
            l2_alpha: Optional[float] = None,
            stochastic: bool = False,
            batch_size: Optional[float] = None
    ):
        super().__init__(max_iter, lr, l1_alpha, l2_alpha, stochastic, batch_size)

    def _calculate_gradient(self, X_mat, y_vec, batch_idxs):
        for i in batch_idxs:
            x_i, y_i = X_mat[i], y_vec[i]
            self._W += self._lr * x_i * (y_i - 1 / (1 + np.exp(-np.dot(x_i, self._W))))

    def _calculate_preds(self, X_mat):
        preds = 1 / (1 + np.exp(-np.dot(X_mat, self._W)))
        return (preds >= 0.5).astype(int)
