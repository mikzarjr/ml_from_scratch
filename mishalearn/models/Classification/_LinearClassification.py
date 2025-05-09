from typing import Union, Optional

import numpy as np
import pandas as pd


class BaseClassifier:

    def __init__(
            self,
            max_iter: int = 1000,
            lr: float = 0.001
    ):
        self._max_iter = max_iter
        self._lr = lr
        self._W: Optional[np.ndarray] = None

    @staticmethod
    def _prepare_data(X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.to_numpy()
        return np.array(X)

    def fit(self,
            X: Union[pd.DataFrame, pd.Series, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> None:
        raise NotImplementedError("Метод fit должен быть реализован в подклассе")

    def predict(self,
                X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.Series:
        X_np = self._prepare_data(X)
        X_mat = np.insert(X_np, 0, 1.0, axis=1)
        preds = np.sign(X_mat.dot(self._W))
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return pd.Series(preds, index=X.index)
        return pd.Series(preds)


class LinearClassification(BaseClassifier):
    # TODO: объединить функции _simple_descend и _stochastic_descend
    def __init__(
            self,
            max_iter: int = 2000,
            lr: float = 0.001,
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

        self._descend_method = (
            self._stochastic_descend if stochastic else self._simple_descend
        )

    def fit(self,
            X: pd.DataFrame | pd.Series | np.ndarray,
            y: pd.Series | np.ndarray) -> None:
        X_np = self._prepare_data(X)
        X_mat = np.insert(X_np, 0, 1.0, axis=1)
        y_vec = y.to_numpy().flatten() if isinstance(y, pd.Series) else np.array(y).flatten()

        self._W = np.random.randn(X_mat.shape[1])
        self._descend_method(X_mat, y_vec)

    def _regularization(self):
        if self._l2_alpha is not None:
            self._W[1:] *= (1 - self._lr * self._l2_alpha)
        if self._l1_alpha is not None:
            self._W[1:] -= self._lr * self._l1_alpha * np.sign(self._W[1:])

    def _simple_descend(self, X_mat: np.ndarray, y_vec: np.ndarray):
        for _ in range(self._max_iter):
            self._regularization()
            for x_i, y_i in zip(X_mat, y_vec):
                if y_i * np.dot(self._W, x_i) <= 0:
                    self._W += self._lr * y_i * x_i

    def _stochastic_descend(self, X_mat: np.ndarray, y_vec: np.ndarray):
        n = X_mat.shape[0]
        if self._batch_size is None:
            bs = 1
        else:
            bs = int(self._batch_size * n) if 0 < self._batch_size < 1 else int(self._batch_size)
        bs = max(1, min(bs, n))

        for _ in range(self._max_iter):
            idxs = np.random.permutation(n)
            for start in range(0, n, bs):
                batch_idx = idxs[start:start + bs]
                self._regularization()
                for i in batch_idx:
                    x_i, y_i = X_mat[i], y_vec[i]
                    if y_i * np.dot(self._W, x_i) <= 0:
                        self._W += self._lr * y_i * x_i


class SVM(BaseClassifier):
    def __init__(self,
                 max_iter: int = 1000,
                 lr: float = 0.001,
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

        for epoch in range(self.epochs):
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

                self._W -= self.lr * grad_w
                self._b -= self.lr * grad_b

    def predict(self,
                X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.Series:
        X_mat = self._prepare_data(X)
        scores = np.dot(X_mat, self._W) + self._b
        preds = np.sign(scores)
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return pd.Series(preds, index=getattr(X, 'index', None))
        else:
            return pd.Series(preds)
