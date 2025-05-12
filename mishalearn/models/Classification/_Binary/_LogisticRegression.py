import numpy as np

from mishalearn.models._Core import BaseBinaryLinearClassifier


class LogisticRegression_BinaryClassificator(BaseBinaryLinearClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _calculate_gradient(self, X_mat, y_vec, batch_idxs):
        for i in batch_idxs:
            x_i, y_i = X_mat[i], y_vec[i]
            self._W += self._lr * x_i * (y_i - 1 / (1 + np.exp(-np.dot(x_i, self._W))))

    def _calculate_preds(self, X_mat):
        preds = 1 / (1 + np.exp(-np.dot(X_mat, self._W)))
        return (preds >= 0.5).astype(int)
