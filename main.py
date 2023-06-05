
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from Regression.main import *
from Losses.losses import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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

    # pf = PolynomialFeatures(degree=2)
    # X_tr = pf.fit_transform(X)
    # lr = LinearRegression()
    # lr.fit(X_tr, y)
    # x_line = np.linspace(np.min(X), np.max(X), X.shape[0]).reshape(-1,1)
    # x_line_tr = pf.transform(x_line)
    # y_line = lr.predict(x_line_tr)
    # plt.scatter(X, y)
    # plt.plot(x_line, y_line);
    # plt.show()

    # linear_reg = LinearRegression(learning_rate=1e-1, loss=MSE())
    # linear_reg.fit(X,y,epochs=5, animate=ANIMATE)

    # ols_reg = OLSRegression()
    # m, b = ols_reg.fit(X, y, animate=ANIMATE)

    # sgd_reg = SGDRegressor(learning_rate=1e-1, loss=MSE())
    # sgd_reg.fit(X,y,epochs=50, animate=ANIMATE)

    # X = 2*np.array(range(0, 11)) + np.random.normal()
    # y = 2*X**2 - 2*X

    # X = X.reshape(-1, 1)
    polynomial = PolynomialRegression(learning_rate=0.1, degree=2, loss=MSE())
    polynomial.fit(X,y,epochs=300, animate=ANIMATE)





# To do:
# 1. Decide how to implement fit and __animated_fit logic using Inheritance
# 2. Implement Lasso, Ridge, ElasticNet Regressions
# future. Implement optimizers: Adam, RMSProp, Nesterov optimizer.