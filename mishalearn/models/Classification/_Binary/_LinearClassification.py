import numpy as np

from mishalearn.models._Core import BaseBinaryLinearClassifier


class Perceptron_BinaryClassificator(BaseBinaryLinearClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _calculate_gradient(self, X_mat, y_vec, batch_idxs):
        for i in batch_idxs:
            x_i, y_i = X_mat[i], y_vec[i]
            if y_i * np.dot(self._W, x_i) <= 0:
                self._W += self._lr * y_i * x_i

    def _calculate_preds(self, X_mat):
        preds = np.sign(np.dot(X_mat, self._W))
        return preds

    def _prepare_classes_forward(self, y: np.ndarray) -> np.ndarray:
        forward_map = {0: -1, 1: +1}
        return np.array([forward_map[label] for label in y])

    def _prepare_classes_backward(self, preds_vec: np.ndarray) -> np.ndarray:
        forward_map = {-1: 0, +1: 1}
        return np.array([forward_map[label] for label in preds_vec])


class SVM_BinaryClassificator(BaseBinaryLinearClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _calculate_gradient(self, X_mat, y_vec, batch_idxs):
        for i in batch_idxs:
            x_i, y_i = X_mat[i], y_vec[i]
            if 1 - y_i * np.dot(self._W, x_i) > 0:
                self._W += self._lr * y_i * x_i

    def _calculate_preds(self, X_mat):
        preds = np.sign(np.dot(X_mat, self._W))
        return preds

    def _prepare_classes_forward(self, y: np.ndarray) -> np.ndarray:
        forward_map = {0: -1, 1: +1}
        return np.array([forward_map[label] for label in y])

    def _prepare_classes_backward(self, preds_vec: np.ndarray) -> np.ndarray:
        forward_map = {-1: 0, +1: 1}
        return np.array([forward_map[label] for label in preds_vec])
