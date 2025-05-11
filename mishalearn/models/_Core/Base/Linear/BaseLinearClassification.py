from abc import ABC, abstractmethod
from typing import Union, Optional

import numpy as np
import pandas as pd

from .BaseLinear import BaseLinear


class BaseLinearClassifier(BaseLinear, ABC):
    def __init__(self):
        super().__init__()

    def fit(self, X: pd.DataFrame | pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> None:
        X_mat = self._prepare_data(X)
        y_vec = self._prepare_labels(y)

        self._W = np.zeros(X_mat.shape[1])

        self._fit(X_mat, y_vec)

    def _fit(self, X_mat: pd.DataFrame | pd.Series | np.ndarray, y_vec: pd.Series | np.ndarray) -> None:
        """
        Model fit. Must be implemented in the child class.
        """
        raise NotImplementedError("Must be implemented in the child class.")

    def predict(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.Series:
        X_mat = self._prepare_data(X)
        preds_raw = self._calculate_preds(X_mat)
        preds = self._prepare_preds(X, preds_raw)

        return preds

    @abstractmethod
    def _calculate_preds(self, X_mat):
        """
        Predictions computation. Must be implemented in the child class.
        """
        raise NotImplementedError("Must be implemented in the child class.")

    @staticmethod
    def _prepare_data(X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return np.insert(X.to_numpy(), 0, 1.0, axis=1)
        return np.insert(np.array(X), 0, 1.0, axis=1)

    def _prepare_labels(self, y: np.ndarray) -> np.ndarray:
        y_vec = y.to_numpy().flatten() if isinstance(y, pd.Series) else np.array(y).flatten()
        y_unique = set(y_vec)

        self._check_classes(y_unique)
        y_vec = self._prepare_classes_forward(y_vec)
        return y_vec

    def _prepare_preds(self, X: Union[pd.DataFrame, pd.Series, np.ndarray], preds_raw: np.ndarray) -> np.ndarray:
        if isinstance(X, (pd.DataFrame, pd.Series)):
            preds_vec = pd.Series(preds_raw, index=X.index)
        else:
            preds_vec = pd.Series(preds_raw)

        preds = self._prepare_classes_backward(preds_vec)
        return preds

    @staticmethod
    def _prepare_classes_forward(y_vec):
        return y_vec

    @staticmethod
    def _prepare_classes_backward(preds_vec):
        return preds_vec

    @staticmethod
    def _check_classes(y_unique):
        """
        Check valid number of classes:
            <2 - error
            2 - binary
            >2 - multiclass
        Must be implemented in the child class.
        """
        raise NotImplementedError("Must be implemented in the child class.")


class BaseBinaryLinearClassifier(BaseLinearClassifier, ABC):
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

    def _fit(self, X_mat: pd.DataFrame | pd.Series | np.ndarray, y_vec: pd.Series | np.ndarray) -> None:
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
                self._calculate_gradient(X_mat, y_vec, batch_idxs)

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

    @abstractmethod
    def _calculate_gradient(self, X_mat, y_vec, batch_idx):
        """
        Gradient computation. Must be implemented in the child class.
        """
        raise NotImplementedError("Must be implemented in the child class.")


class BaseMultiLinearClassifier(BaseLinearClassifier, ABC):
    def __init__(self):
        super().__init__()
        self._classes = None

    def _fit(self, X_mat: Union[pd.DataFrame, pd.Series, np.ndarray], y_vec: Union[pd.Series, np.ndarray]) -> None:
        self._classes = np.unique(y_vec)
        for cls in self._classes:
            self._fit_multiclass(X_mat, y_vec, cls)

    @abstractmethod
    def _fit_multiclass(self, X_mat: np.ndarray, y_arr: np.ndarray, cls: int) -> None:
        """
        Multiclass fit. Must be implemented in the child class.
        """
        raise NotImplementedError("Must be implemented in the child class.")

    @staticmethod
    def _check_classes(y_unique):
        if len(y_unique) <= 2:
            raise ValueError("Number of target classes must be > 2 for multiclass classification")
        if min(y_unique) != 0 and max(y_unique) != len(y_unique) - 1:
            raise ValueError("Target classes must be formatted as 0, 1, 2...")
        return
