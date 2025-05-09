from abc import ABC, abstractmethod
from typing import Union, Optional

import numpy as np
import pandas as pd


class BaseClassifier(ABC):
    def __init__(
            self,
            max_iter: int,
            lr: float,
            l1_alpha: Optional[float],
            l2_alpha: Optional[float],
            stochastic: bool,
            batch_size: Optional[float]
    ):
        self._max_iter = max_iter
        self._lr = lr
        self._l1_alpha = l1_alpha
        self._l2_alpha = l2_alpha
        self._stochastic = stochastic
        self._stochastic = stochastic
        self._batch_size = batch_size
        self._W: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame | pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> None:
        X_np = self._prepare_data(X)
        X_mat = np.insert(X_np, 0, 1.0, axis=1)
        y_raw = y.to_numpy().flatten() if isinstance(y, pd.Series) else np.array(y).flatten()
        y_vec = self._prepare_labels(y_raw)

        self._W = np.zeros(X_mat.shape[1])

        n = X_mat.shape[0]
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
                self._compute_gradient(X_mat, y_vec, batch_idxs)

    def predict(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.Series:
        X_np = self._prepare_data(X)
        X_mat = np.insert(X_np, 0, 1.0, axis=1)
        preds = self._calculate_preds(X_mat)
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return pd.Series(preds, index=X.index)
        return pd.Series(preds)

    @abstractmethod
    def _calculate_preds(self, X_mat):
        """
        Predictions computation. Must be implemented in the child class.
        """
        raise NotImplementedError("Must be implemented in the child class.")

    @abstractmethod
    def _compute_gradient(self, X_mat, y_vec, batch_idx):
        """
        Gradient computation. Must be implemented in the child class.
        """
        raise NotImplementedError("Must be implemented in the child class.")

    def _regularization(self):
        if self._l2_alpha is not None:
            self._W[1:] *= (1 - self._lr * self._l2_alpha)
        if self._l1_alpha is not None:
            self._W[1:] -= self._lr * self._l1_alpha * np.sign(self._W[1:])

    @staticmethod
    def _prepare_data(X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.to_numpy()
        return np.array(X)

    @staticmethod
    def _prepare_labels(y_vec: np.ndarray) -> np.ndarray:
        return y_vec


class LinearClassification(BaseClassifier):
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

    def _calculate_preds(self, X_mat):
        preds = np.sign(np.dot(X_mat, self._W))
        return ((preds + 1) // 2).astype(int)

    def _compute_gradient(self, X_mat, y_vec, batch_idxs):
        for i in batch_idxs:
            x_i, y_i = X_mat[i], y_vec[i]
            if y_i * np.dot(self._W, x_i) <= 0:
                self._W += self._lr * y_i * x_i

    def _prepare_labels(self, y_vec: np.ndarray) -> np.ndarray:
        return np.where(y_vec == 0, -1, +1)


class SVM(BaseClassifier):
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

    def _compute_gradient(self, X_mat, y_vec, batch_idxs):
        for i in batch_idxs:
            x_i, y_i = X_mat[i], y_vec[i]
            if 1 - y_i * np.dot(self._W, x_i) > 0:
                self._W += self._lr * y_i * x_i

    def _calculate_preds(self, X_mat):
        preds = np.sign(np.dot(X_mat, self._W))
        return ((preds + 1) // 2).astype(int)

    def _prepare_labels(self, y_vec: np.ndarray) -> np.ndarray:
        return np.where(y_vec == 0, -1, +1)


class LogisticRegression(BaseClassifier):
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

    def _compute_gradient(self, X_mat, y_vec, batch_idxs):
        for i in batch_idxs:
            x_i, y_i = X_mat[i], y_vec[i]
            self._W += self._lr * x_i * (y_i - 1 / (1 + np.exp(-np.dot(x_i, self._W))))

    def _calculate_preds(self, X_mat):
        preds = 1 / (1 + np.exp(-np.dot(X_mat, self._W)))
        return (preds >= 0.5).astype(int)
