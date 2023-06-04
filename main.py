
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

from Regression.main import *
from Losses.losses import *

np.random.seed(1)
# plt.style.use('fivethirtyeight')
class StopRunError(Exception):
    pass

# plt.grid(True)
if __name__=='__main__':
    N_FEATURES = 1
    ANIMATE = True
    assert (not ANIMATE and N_FEATURES==1) or N_FEATURES==1, 'Cannot animate more than 1 dimension'

    X, y = make_regression(n_features=N_FEATURES, noise=15)
    X = np.append(X, 10).reshape(-1, N_FEATURES) # add an outlier for Ridge and Lasso regressions
    y = np.append(y, 100) 
    # linear_reg = LinearRegression(learning_rate=1e-1, loss=MSE())
    # linear_reg.fit(X,y,epochs=5, animate=ANIMATE)

    # ols_reg = OLSRegression()
    # m, b = ols_reg.fit(X, y, animate=ANIMATE)

    # sgd_reg = SGDRegressor(learning_rate=1e-1, loss=MSE())
    # sgd_reg.fit(X,y,epochs=50, animate=ANIMATE)


    polynomial = PolynomialRegression(learning_rate=0.01, degree=2, loss=MSE())
    polynomial.fit(X,y,epochs=500, animate=ANIMATE)

# To do:
# 1. Decide how to implement fit and __animated_fit logic using Inheritance
# 2. Implement Lasso, Ridge, ElasticNet Regressions
# future. Implement optimizers: Adam, RMSProp, Nesterov optimizer.