from abc import ABC, abstractmethod
from typing import Union, Optional

import numpy as np
import pandas as pd


class BaseBinClassifier(ABC):
    def __init__(
            self,
            max_iter: int = 1000,
            lr: float = 0.001
    ):
        self._max_iter = max_iter
        self._lr = lr
        self._W: Optional[np.ndarray] = None
        self._b: float = 0.0

    @abstractmethod
    def fit(self,
            X: Union[pd.DataFrame, pd.Series, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> None:
        """
        Model fit. Must be implemented in the child class.
        """
        raise NotImplementedError

    def predict(self,
                X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.Series:
        """
        Predict (+1 / -1).
        """
        X_mat = self._prepare_data(X)
        scores = np.dot(X_mat, self._W) + self._b
        preds = np.sign(scores)
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return pd.Series(preds, index=getattr(X, 'index', None))
        else:
            return pd.Series(preds)

    @staticmethod
    def _prepare_data(X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.to_numpy()
        return np.array(X)


class SVM(BaseBinClassifier):
    def __init__(self,
                 max_iter: int = None,
                 lr: float = None,
                 lambda_param: float = 0.01,
                 stochastic: bool = True):
        super().__init__(max_iter, lr)
        self._lambda_param = lambda_param
        self._stochastic = stochastic

    def fit(self,
            X: Union[pd.DataFrame, pd.Series, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> None:
        X_mat = self._prepare_data(X)
        y_vec = y.to_numpy().flatten() if isinstance(y, pd.Series) else np.array(y).flatten()

        n_samples, n_features = X_mat.shape
        self._W = np.zeros(n_features)
        self._b = 0.0

        for epoch in range(self._max_iter):
            idxs = np.random.permutation(n_samples) if self._stochastic else np.arange(n_samples)

            for i in idxs:
                x_i = X_mat[i]
                y_i = y_vec[i]
                condition = y_i * (np.dot(self._W, x_i) + self._b)

                if condition < 1:
                    grad_w = self._lambda_param * self._W - y_i * x_i
                    grad_b = -y_i
                else:
                    grad_w = self._lambda_param * self._W
                    grad_b = 0.0

                self._W -= self._lr * grad_w
                self._b -= self._lr * grad_b


class LinearClassification(BaseBinClassifier):
    def __init__(
            self,
            max_iter: int = None,
            lr: float = None,
            l1_alpha: Optional[float] = None,
            l2_alpha: Optional[float] = None,
            stochastic: bool = False,
            batch_size: Optional[float] = None
    ):
        super().__init__(max_iter, lr)
        self._l1_alpha = l1_alpha
        self._l2_alpha = l2_alpha
        self._stochastic = stochastic
        self._batch_size = batch_size

    def _regularize(self):
        if self._l2_alpha is not None:
            self._W *= (1 - self._lr * self._l2_alpha)
        if self._l1_alpha is not None:
            self._W -= self._lr * self._l1_alpha * np.sign(self._W)

    def fit(self,
            X: Union[pd.DataFrame, pd.Series, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> None:
        X_mat = self._prepare_data(X)
        n_samples, n_features = X_mat.shape

        self._W = np.zeros(n_features)
        self._b = 0.0

        if self._batch_size is None:
            bs = 1 if self._stochastic else n_samples
        else:
            if 0 < self._batch_size < 1:
                bs = int(self._batch_size * n_samples)
            else:
                bs = int(self._batch_size)
            bs = max(1, min(bs, n_samples))

        y_vec = y.to_numpy().flatten() if isinstance(y, pd.Series) else np.array(y).flatten()

        for epoch in range(self._max_iter):
            idxs = np.random.permutation(n_samples) if self._stochastic else np.arange(n_samples)

            for start in range(0, n_samples, bs):
                batch_idx = idxs[start:start + bs]

                self._regularize()

                for i in batch_idx:
                    x_i = X_mat[i]
                    y_i = y_vec[i]
                    margin = y_i * (np.dot(self._W, x_i) + self._b)
                    if margin <= 0:
                        self._W += self._lr * y_i * x_i
                        self._b += self._lr * y_i
