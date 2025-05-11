__ALL__ = [
    'OLS_Regression',
    'Linear_Regression',
    'Ridge_Regression',
    'Lasso_Regression',
    'ElasticNet_Regression'
]

from ._LinearRegression import (
    Linear_Regression,
    Ridge_Regression,
    Lasso_Regression,
    ElasticNet_Regression
)
from ._OrdinaryLeastSquares import OLS_Regression
