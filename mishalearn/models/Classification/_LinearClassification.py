from typing import Optional

import numpy as np
import pandas as pd


class LinearClassification:
    def __init__(self,
                 max_iter: int = 1000,
                 lr: float = 0.001,
                 l1_alpha: Optional[float] = None,
                 l2_alpha: Optional[float] = None,
                 stochastic: bool = False,
                 batch_size: int | float = None):

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
        if self._batch_size is None:
            bs = 1
        else:
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
