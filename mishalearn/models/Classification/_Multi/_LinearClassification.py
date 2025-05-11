from abc import ABC, abstractmethod
from typing import Union, Any, Dict

import numpy as np
import pandas as pd

from mishalearn.models.Classification.BaseLC import BaseClassifier


class BaseMultiClassifier(BaseClassifier, ABC):
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


class OneVsAllClassifier(BaseMultiClassifier, ABC):
    def __init__(self,
                 base_clf_cls: Any,
                 **base_clf_params):
        """
        _base_clf_cls    — класс бинарного классификатора
                          (например, LinearClassification или SVM)
        base_clf_params — его конструктор-параметры
                          (lr, max_iter, l1_alpha, l2_alpha, и т.д.)
        """
        super().__init__()
        self._base_clf_cls = base_clf_cls
        self._base_clf_params = base_clf_params
        self._clfs: Dict[Any, BaseClassifier] = {}

    def _fit_multiclass(self, X_mat: np.ndarray, y_arr: np.ndarray, cls: int) -> None:
        y_bin = (y_arr == cls).astype(int)
        clf = self._base_clf_cls(**self._base_clf_params)
        clf.fit(X_mat[:, 1:], pd.Series(y_bin))
        self._clfs[cls] = clf

    def _calculate_preds(self, X_mat: np.ndarray) -> np.ndarray:
        scores = []
        for c in self._classes:
            clf = self._clfs[c]
            scores.append(X_mat.dot(clf._W))

        scores_mat = np.vstack(scores).T
        idxs = np.argmax(scores_mat, axis=1)
        return self._classes[idxs]


class AllVsAllClassifier(BaseMultiClassifier):
    def __init__(self,
                 base_clf_cls: Any,
                 **base_clf_params):
        super().__init__()
        self._base_clf_cls = base_clf_cls
        self._base_clf_params = base_clf_params
        self._clfs: Dict[Any, BaseClassifier] = {}

    def _fit_multiclass(self,
                        X_mat: np.ndarray,
                        y_arr: np.ndarray,
                        cls: int) -> None:

        for other in self._classes:
            if other <= cls:
                continue

            mask = (y_arr == cls) | (y_arr == other)
            X_pair = X_mat[mask]
            y_pair = y_arr[mask]

            y_bin = (y_pair == cls).astype(int)
            clf = self._base_clf_cls(**self._base_clf_params)
            clf.fit(X_pair[:, 1:], y_bin)
            self._clfs[(cls, other)] = clf

    def _calculate_preds(self, X_mat: np.ndarray) -> np.ndarray:
        n = X_mat.shape[0]
        K = len(self._classes)
        votes = np.zeros((n, K), dtype=int)

        X_raw = X_mat[:, 1:]

        for (ci, cj), clf in self._clfs.items():
            y_pred = clf.predict(X_raw)
            for i, pred in enumerate(y_pred):
                winner = ci if pred == 1 else cj
                class_idx = int(np.where(self._classes == winner)[0])
                votes[i, class_idx] += 1

        winner_idxs = np.argmax(votes, axis=1)
        return self._classes[winner_idxs]
