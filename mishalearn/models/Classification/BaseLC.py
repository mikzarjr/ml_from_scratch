from abc import ABC, abstractmethod
from typing import Union, Optional

import numpy as np
import pandas as pd


class BaseClassifier(ABC):
    def __init__(self):
        self._W: Optional[np.ndarray] = None

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
