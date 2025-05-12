from typing import Dict, Any

import numpy as np

from mishalearn.models.Classification._Binary._LogisticRegression import LogisticRegression_BinaryClassificator
from mishalearn.models._Core import BaseMultiLinearClassifier


class LogisticRegression_MultiClassificator(BaseMultiLinearClassifier):
    def __init__(self, **LogRegparams):
        """
        LogRegparams — Logistic Regressor parameters:
                    (lr, max_iter, l1_alpha, l2_alpha, etc.)
        """
        super().__init__()
        self._params: Dict[str, Any] = LogRegparams
        self._clfs: Dict[Any, LogisticRegression_BinaryClassificator] = {}

    def _fit_multiclass(self, X_mat: np.ndarray, y_arr: np.ndarray, cls: int) -> None:
        y_binary = (y_arr == cls).astype(int)

        clf = LogisticRegression_BinaryClassificator(**self._params)
        clf._W = np.zeros(X_mat.shape[1])
        clf._fit(X_mat, y_binary)
        self._clfs[cls] = clf

    def _calculate_preds(self, X_mat: np.ndarray) -> np.ndarray:
        n_samples = X_mat.shape[0]
        n_classes = len(self._classes)
        probs = np.zeros((n_samples, n_classes))

        sigmoid = lambda z: 1 / (1 + np.exp(-z))

        for idx, cls in enumerate(self._classes):
            clf = self._clfs[cls]
            z = X_mat.dot(clf._W)
            probs[:, idx] = sigmoid(z)

        preds_idx = np.argmax(probs, axis=1)
        return self._classes[preds_idx]
