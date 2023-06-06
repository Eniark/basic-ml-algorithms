
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from Regression.regressions import *
from Losses.losses import *
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Lasso, SGDRegressor
# from sklearn.preprocessing import StandardScaler
np.random.seed(1)
# plt.style.use('fivethirtyeight')

if __name__=='__main__':
    N_FEATURES = 1
    ANIMATE = True
    assert (not ANIMATE and N_FEATURES==1) or N_FEATURES==1, 'Cannot animate more than 1 dimension'

    X, y = make_regression(n_features=N_FEATURES, noise=15)
    X = np.array([[0.05, 0.1,0.5,0.51,0.51,0.52,0.8, 0.9, 1.2, 1.3,1.55,1.75,1.76,2.1,2.25,2.4,2.5,2.8,2.9, 2.9]]).reshape(-1,1)
    y = np.array([0.25,1.1,1.05,2.25,0.8,1.1,2.5,0.6,1.4,1.2,2.2,1.05,0.95,1.5,2,1.5,1.85,1.4,2.5,2.6])
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

    linear_reg = LinearRegression(learning_rate=1e-1, loss=MSE())
    linear_reg.fit(X,y,epochs=50, animate=ANIMATE)

    # ols_reg = OLSRegression()
    # m, b = ols_reg.fit(X, y, animate=ANIMATE)

    # sgd_reg = SGDRegressor(learning_rate=1e-1, loss=MSE())
    # sgd_reg.fit(X,y,epochs=50, animate=ANIMATE)

    # X = 2*np.array(range(0, 11)) + np.random.normal()
    # y = 2*X**2 - 2*X

    # X = X.reshape(-1, 1)
    # polynomial = PolynomialRegression(learning_rate=0.01, degree=3, loss=MSE())
    # polynomial.fit(X,y,epochs=500, animate=ANIMATE)
    # sc = StandardScaler()

    # X = sc.fit_transform(X,y)
    # l = Lasso(alpha=0.9)

    # l.fit(X,y)

    # m = Lasso(alpha=0.1)
    # m.fit(X,y)
    # print(m.coef_, m.intercept_)



    # b, m =  m.intercept_, m.coef_[0]
    # x_line = np.linspace(np.min(X), np.max(X),20)
    # y_line = Regression.get_line(x_line, m, b)

    # plt.scatter(X, y)
    # plt.plot(x_line, y_line);
    # plt.show()
    # print(l.intercept_)
    # X = sc.fit_transform(X, y)
    # lasso_reg = LassoRegression(alpha=0.9, learning_rate=1e-1, loss=MSE())
    # lasso_reg.fit(X,y,epochs=100, animate=ANIMATE)
    # print(linear_reg.W)





# NOTES:
# 1. Implement Lasso, Ridge, ElasticNet Regressions
# 2. Regression using normal equation?
# 3. Create metrics file(or folder) for r_squared
# 4. KNN.





# future. Implement optimizers: Adam, RMSProp, Nesterov optimizer.