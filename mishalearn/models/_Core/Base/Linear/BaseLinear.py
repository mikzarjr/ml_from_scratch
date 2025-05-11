from abc import ABC
from typing import Union, Optional

import numpy as np
import pandas as pd


class BaseLinear(ABC):
    def __init__(self):
        self._W: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame | pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> None:
        """
        Model fit. Must be implemented in the child class.
        """
        raise NotImplementedError("Must be implemented in the child class.")

    def predict(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.Series:
        """
        Predictions computation. Must be implemented in the child class.
        """
        raise NotImplementedError("Must be implemented in the child class.")
