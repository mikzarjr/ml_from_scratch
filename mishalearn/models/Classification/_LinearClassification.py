from abc import ABC, abstractmethod
from typing import Union, Optional

import numpy as np
import pandas as pd


class BaseClassifier(ABC):
    def __init__(self):
        self.W: Optional[np.ndarray] = None
        self.b: float = 0.0

    @abstractmethod
    def fit(self,
            X: Union[pd.DataFrame, pd.Series, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> None:
        """
        Model fit. Must be implemented in the child class.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self,
                X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.Series:
        """
        Predict (+1 / -1). Must be implemented in the child class.
        """
        raise NotImplementedError

    def _prepare_data(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.to_numpy()
        return np.array(X)


class SVM(BaseClassifier):
    def __init__(self,
                 lr: float = 0.001,
                 lambda_param: float = 0.01,
                 epochs: int = 1000,
                 stochastic: bool = True):
        super().__init__()
        self.lr = lr
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.stochastic = stochastic

    def fit(self,
            X: Union[pd.DataFrame, pd.Series, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> None:
        X_mat = self._prepare_data(X)
        y_vec = y.to_numpy().flatten() if isinstance(y, pd.Series) else np.array(y).flatten()

        n_samples, n_features = X_mat.shape
        self.W = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.epochs):
            idxs = np.random.permutation(n_samples) if self.stochastic else np.arange(n_samples)

            for i in idxs:
                x_i = X_mat[i]
                y_i = y_vec[i]
                condition = y_i * (np.dot(self.W, x_i) + self.b)

                if condition < 1:
                    grad_w = self.lambda_param * self.W - y_i * x_i
                    grad_b = -y_i
                else:
                    grad_w = self.lambda_param * self.W
                    grad_b = 0.0

                self.W -= self.lr * grad_w
                self.b -= self.lr * grad_b

    def predict(self,
                X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.Series:
        X_mat = self._prepare_data(X)
        scores = np.dot(X_mat, self.W) + self.b
        preds = np.sign(scores)
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return pd.Series(preds, index=getattr(X, 'index', None))
        else:
            return pd.Series(preds)


class LinearClassification(BaseClassifier):
    def __init__(self,
                 max_iter: int = 1000,
                 lr: float = 0.001,
                 l1_alpha: Optional[float] = None,
                 l2_alpha: Optional[float] = None,
                 stochastic: bool = False,
                 batch_size: Optional[float] = None):
        super().__init__()
        self.max_iter = max_iter
        self.lr = lr
        self.l1_alpha = l1_alpha
        self.l2_alpha = l2_alpha
        self.stochastic = stochastic
        self.batch_size = batch_size

    def _regularize(self):
        if self.l2_alpha is not None:
            self.W *= (1 - self.lr * self.l2_alpha)
        if self.l1_alpha is not None:
            self.W -= self.lr * self.l1_alpha * np.sign(self.W)

    def fit(self,
            X: Union[pd.DataFrame, pd.Series, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> None:
        X_mat = self._prepare_data(X)
        n_samples, n_features = X_mat.shape

        self.W = np.zeros(n_features)
        self.b = 0.0

        if self.batch_size is None:
            bs = 1 if self.stochastic else n_samples
        else:
            if 0 < self.batch_size < 1:
                bs = int(self.batch_size * n_samples)
            else:
                bs = int(self.batch_size)
            bs = max(1, min(bs, n_samples))

        y_vec = y.to_numpy().flatten() if isinstance(y, pd.Series) else np.array(y).flatten()

        for epoch in range(self.max_iter):
            idxs = np.random.permutation(n_samples) if self.stochastic else np.arange(n_samples)

            for start in range(0, n_samples, bs):
                batch_idx = idxs[start:start + bs]

                self._regularize()

                for i in batch_idx:
                    x_i = X_mat[i]
                    y_i = y_vec[i]
                    margin = y_i * (np.dot(self.W, x_i) + self.b)
                    if margin <= 0:
                        self.W += self.lr * y_i * x_i
                        self.b += self.lr * y_i

    def predict(self,
                X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.Series:
        X_mat = self._prepare_data(X)
        scores = X_mat.dot(self.W) + self.b
        preds = np.sign(scores)
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return pd.Series(preds, index=X.index)
        return pd.Series(preds)


class LinearClassification:
    def __init__(self,
                 max_iter: int = 1000,
                 lr: float = 0.001,
                 l1_alpha: Optional[float] = None,
                 l2_alpha: Optional[float] = None,
                 stochastic: bool = False,
                 batch_size: int | float = 1):

        self._W = None
        self._max_iter = max_iter
        self._lr = lr
        self._l1_alpha = l1_alpha
        self._l2_alpha = l2_alpha

        self._descend_method = (
            self._stochastic_descend if stochastic else self._simple_descend
        )
        self._batch_size = batch_size

    def fit(self, X: pd.DataFrame | pd.Series, y: pd.Series):
        X_mat = X.copy().to_numpy()
        X_mat = np.insert(X_mat, 0, 1.0, axis=1)
        y_vec = y.to_numpy().flatten()
        self._W = np.random.randn(X_mat.shape[1])
        self._descend_method(X_mat, y_vec)

    def _regularization(self):
        if self._l2_alpha is not None:
            self._W[1:] *= (1 - self._lr * self._l2_alpha)
        if self._l1_alpha is not None:
            self._W[1:] -= self._lr * self._l1_alpha * np.sign(self._W[1:])

    def _simple_descend(self, X_mat, y_vec):
        for epoch in range(self._max_iter):
            self._regularization()

            for X_i, y_i in zip(X_mat, y_vec):
                margin = y_i * np.dot(self._W, X_i)
                if margin <= 0:
                    self._W += self._lr * y_i * X_i

    def _stochastic_descend(self, X_mat, y_vec):
        n = X_mat.shape[0]
        bs = int(self._batch_size * n) if self._batch_size < 1 else int(self._batch_size)
        bs = max(1, min(bs, n))

        for epoch in range(self._max_iter):
            idxs = np.random.permutation(n)
            for start in range(0, n, bs):
                batch_idx = idxs[start:start + bs]
                X_batch = X_mat[batch_idx]
                y_batch = y_vec[batch_idx]

                self._regularization()

                for X_i, y_i in zip(X_batch, y_batch):
                    margin = y_i * np.dot(self._W, X_i)
                    if margin <= 0:
                        self._W += self._lr * y_i * X_i

    def predict(self, X: pd.DataFrame | pd.Series) -> pd.Series:
        X_mat = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_mat.insert(0, 'Intercept', 1.0)
        preds = np.sign(np.dot(X_mat, self._W))
        return pd.Series(preds, index=X_mat.index)
