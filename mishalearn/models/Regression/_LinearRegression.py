import os
import shutil

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from mishalearn.metrics import mean_squared_error
from abc import ABC, abstractmethod


class OLSRegression:

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


class BaseLinearModel(ABC):
    def __init__(
            self,
            lr: float = 0.001,
            delta_converged: float = 1e-6,
            max_steps: int = 10000,
            batch_size: int | float = 0.01,
            loss: callable = mean_squared_error,
            descend: str = 'simple',
            verbose: bool = False,
            show_graph: bool = False,
    ):
        """
        Parameters
        ----------
        lr : float
            Learning rate.
        delta_converged : float
            Convergence threshold for early stopping.
        max_steps : int
            Maximum number of optimization steps.
        batch_size : int or float
            Size of mini-batch (absolute or relative if <1).
        loss : callable
            Loss function (default: mean_squared_error).
        descend : str
            'simple' for full-batch gradient descent, 'stochastic' for SGD.
        verbose : bool
            Whether to print training logs.
        show_graph : bool
            Whether to plot and save loss graph after training.
        """

        self._lr = lr
        self._delta = delta_converged
        self._max_steps = max_steps
        self._batch_size = batch_size
        self._loss = loss
        self._verbose = verbose
        self._show_graph = show_graph

        if self._show_graph:
            self._graph_initialized = False

        descend_methods = {
            'stochastic': self._stochastic_gradient_descent,
            'simple': self._gradient_descent
        }
        self._descend_method = descend_methods[descend]

        self._w = None
        self._X = None
        self._y = None
        self._cost_history = []

    def fit(self, X: pd.DataFrame | pd.Series, y: pd.Series) -> None:
        self._cost_history = []
        X_ = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_.insert(0, 'Intercept', 1.0)
        self._X = X_.to_numpy()
        self._y = y.to_numpy().flatten()
        self._w = np.random.randn(self._X.shape[1])

        self._descend_method()

    def predict(self, X: pd.DataFrame | pd.Series) -> pd.Series:
        X_ = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_.insert(0, 'Intercept', 1.0)
        preds = np.dot(X_, self._w)
        return pd.Series(preds, index=X_.index)

    def _stochastic_gradient_descent(self):
        m = self._X.shape[0]
        if self._batch_size < 1:
            self._batch_size = int(self._batch_size * m)

        prev_loss = float('inf')

        for step in range(self._max_steps):
            indices = np.random.permutation(m)
            X_shuffled = self._X[indices]
            y_shuffled = self._y[indices]

            start = 0
            X_batch = X_shuffled[start:start + self._batch_size]
            y_batch = y_shuffled[start:start + self._batch_size]

            preds = np.dot(X_batch, self._w)
            err = preds - y_batch
            grad = self._compute_gradient(X_batch, err)

            self._w -= self._lr * grad

            full_preds = self._X.dot(self._w)
            cost = self._loss(self._y, full_preds)
            self._cost_history.append(cost)

            if self._verbose and step % 100 == 0:
                print(f"Step {step}, Loss: {cost:.6f}")

            if abs(prev_loss - cost) < self._delta:
                if self._verbose:
                    print(f"Converged at step {step}")
                break
            prev_loss = cost

        if self._verbose:
            self._plot_loss()

    def _gradient_descent(self):
        prev_loss = float('inf')

        for step in range(self._max_steps):
            preds = np.dot(self._X, self._w)
            err = preds - self._y
            grad = self._compute_gradient(self._X, err)

            self._w -= self._lr * grad

            curr_loss = self._loss(self._y, preds)
            self._cost_history.append(curr_loss)

            if self._verbose and step % 100 == 0:
                print(f"Step {step}, Loss: {curr_loss:.6f}")

            if abs(curr_loss - prev_loss) < self._delta:
                if self._verbose:
                    print(f"Converged at step {step}")
                break
            prev_loss = curr_loss

        if self._verbose:
            self._plot_loss()

    @abstractmethod
    def _compute_gradient(self, X_batch, err):
        """
        Gradient computation. Must be implemented in the child class.
        """
        raise NotImplementedError("Must be implemented in the child class.")

    def _plot_loss(self) -> None:
        if not self._show_graph or not self._cost_history:
            return

        cwd = os.getcwd()
        graphs_dir = f"{cwd}/linreg_graphs"
        if not self._graph_initialized:
            if os.path.exists(graphs_dir):
                shutil.rmtree(graphs_dir)
            os.makedirs(graphs_dir, exist_ok=True)
            self._graph_initialized = True

        i = 1
        while os.path.exists(f"{graphs_dir}/loss_plot_{i}.png"):
            i += 1
        filename = f"{graphs_dir}/loss_plot_{i}.png"

        plt.figure(figsize=(8, 4))
        plt.plot(self._cost_history, label="Loss", linewidth=2, color="#007acc")
        plt.xlabel("Итерация")
        plt.ylabel("Loss (MSE)")
        desc_type = "SGD" if self._descend_method == self._stochastic_gradient_descent else "GD"
        plt.title(f"Сходимость ({desc_type}) — Loss (MSE)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


class LinearRegression(BaseLinearModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_gradient(self, X, err):
        return 2 / X.shape[0] * np.dot(X.T, err)


class RidgeRegression(BaseLinearModel):
    def __init__(self, alpha=0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def _compute_gradient(self, X_batch, err):
        reg_vector = np.concatenate([[0], self._w[1:]])
        return 2 / X_batch.shape[0] * np.dot(X_batch.T, err) + 2 * self.alpha * reg_vector


class LassoRegression(BaseLinearModel):
    def __init__(self, alpha=0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def _compute_gradient(self, X_batch, err):
        reg_vector = np.concatenate([[0], np.sign(self._w[1:])])
        return 2 / X_batch.shape[0] * np.dot(X_batch.T, err) + self.alpha * reg_vector


class ElasticNetRegression(BaseLinearModel):
    def __init__(self, alpha1=0.01, alpha2=0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def _compute_gradient(self, X_batch, err):
        l1_reg_vector = np.concatenate([[0], np.sign(self._w[1:])])
        l2_reg_vector = np.concatenate([[0], self._w[1:]])
        regularization_term = self.alpha1 * l1_reg_vector + 2 * self.alpha2 * l2_reg_vector
        return 2 / X_batch.shape[0] * np.dot(X_batch.T, err) + regularization_term
