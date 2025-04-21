import numpy as np
import pandas as pd

from mishalearn.metrics import mean_squared_error


class OLS_Regression:
    def __init__(self):
        self.w: pd.Series = None

    def fit(self, X: pd.DataFrame | pd.Series, y: pd.Series) -> None:
        """
        :param X: Features DataFrame
        :param y: Target Series
        :return: self
        """
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_df.insert(0, 'Intercept', 1.0)
        X_df_t = X_df.T
        xtx_inv = np.linalg.inv(np.dot(X_df_t, X_df))
        xty = np.dot(X_df_t, y)
        w = np.dot(xtx_inv, xty)
        self.w = pd.Series(w, index=X_df.columns)
        return self

    def predict(self, X: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """
        :param X: Features DataFrame
        :return: Target Series
        """
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_df.insert(0, 'Intercept', 1.0)
        predictions = np.dot(X_df, self.w)
        return predictions


class LIN_Regression:
    def __init__(
            self,
            alpha: float = 1e-3,
            tolerance: float = 1e-6,
            max_iter: int = 10000,
            loss: callable = mean_squared_error,
            descend: str = 'simple'
    ):
        self.__alpha: float = alpha
        self.__tolerance: float = tolerance
        self.__max_iter: int = max_iter
        self.__loss: callable = loss

        descend_methods = {
            'stocastic': self.__stochastic_gradient_descent,
            'simple': self.__gradient_descent
        }
        self.descend_method = descend_methods[descend]

        self.__w: pd.Series = None
        self.__y: pd.Series = None
        self.__X: pd.DataFrame = None

    def fit(self, X: pd.DataFrame | pd.Series, y: pd.Series) -> None:
        """
        :param X: Features DataFrame
        :param y: Target Series
        :return: self
        """
        X_ = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_.insert(0, 'Intercept', 1.0)
        self.__X = X_.to_numpy()
        self.__y = y.to_numpy().flatten()

        self.__w = np.zeros(self.__X.shape[1])
        self.descend_method()

    def __gradient_descent(self):
        prev_loss = float('inf')

        for epoch in range(self.__max_iter):
            self.__update_weights()
            pred = np.dot(self.__X, self.__w)
            curr_loss = self.__loss(self.__y, pred)

            if abs(curr_loss - prev_loss) < self.__tolerance:
                break
            prev_loss = curr_loss

    def __update_weights(self) -> None:
        preds = np.dot(self.__X, self.__w)
        err = preds - self.__y
        grad = 2 * np.dot(self.__X.T, err) / self.__X.shape[0]
        self.__w -= self.__alpha * grad

    def __stochastic_gradient_descent(self, batch_size: int = 10):
        prev_loss = float('inf')
        n_samples = self.__X.shape[0]

        for epoch in range(self.__max_iter):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start:start + batch_size]
                X_batch = self.__X[batch_idx]
                y_batch = self.__y[batch_idx]
                self.__stochastic_update_weights(X_batch, y_batch)
            preds = self.__X.dot(self.__w)
            curr_loss = self.__loss(self.__y.flatten(), preds.flatten())
            if abs(prev_loss - curr_loss) < self.__tolerance:
                break
            prev_loss = curr_loss

    def __stochastic_update_weights(self, X_batch: np.ndarray, y_batch: np.ndarray):
        preds = np.dot(X_batch, self.__w)
        err = preds - y_batch
        grad = 2 * np.dot(X_batch.T, err) / self.__X.shape[0]
        self.__w -= self.__alpha * grad

    def predict(self, X: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """
        :param X: Features DataFrame
        :return: Target Series
        """
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_df.insert(0, 'Intercept', 1.0)
        predictions = np.dot(X_df, self.__w)
        return predictions
