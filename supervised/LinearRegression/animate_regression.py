
from sklearn.datasets import make_regression
from mllib.supervised.LinearRegression.regressions import *
from mllib.utilities.losses import *
from mllib.utilities.lr_schedulers import *
np.random.seed(1)
plt.rcParams['axes.grid'] = True

if __name__=='__main__':
    N_FEATURES = 1
    ANIMATE = True
    LEARNING_RATE = 1e-1
    assert (not ANIMATE and N_FEATURES>=1) or (ANIMATE and N_FEATURES==1), 'Cannot animate more than 1 dimension'

    X, y = make_regression(n_features=N_FEATURES, noise=15)
    X = np.array([[0.05, 0.1,0.5,0.51,0.51,0.52,0.8, 0.9, 1.2, 1.3,1.55,1.75,1.76,2.1,2.25,2.4,2.5,2.8,2.9, 2.9]]).reshape(-1,1)
    y = np.array([0.25,1.1,1.05,2.25,0.8,1.1,2.5,0.6,1.4,1.2,2.2,1.05,0.95,1.5,2,1.5,1.85,1.4,2.5,2.6])

    # X = 2*np.array(range(0, 11)) + np.random.normal()
    # X = X.reshape(-1,1)
    # y = 2*X**2 - 2*X
    # y = y.flatten()
    # sc = StandardScaler()
    # X = sc.fit_transform(X)
    scheduler = configure_power_scheduler(lr0=LEARNING_RATE, every=30)
    # scheduler = configure_exponential_decay(lr0=LEARNING_RATE, every=50)

    # lr = LinearRegression(loss=MSE, learning_rate=LEARNING_RATE, scheduler_fn=None)
    # lr.fit(X, y, epochs=30,animate=ANIMATE)

    lr = LassoRegression(loss=MSE, alpha=0.9, learning_rate=LEARNING_RATE, scheduler_fn=None)
    lr.fit(X, y, epochs=30,animate=ANIMATE)

    # lasso = Lasso(alpha=0.9)
    # lr = PolynomialRegression(loss=MSE,  learning_rate=LEARNING_RATE, scheduler_fn=scheduler, degree=6, penalty=lasso)
    # lr.fit(X, y, epochs=200,animate=ANIMATE)


    # nr = NormalEquation(alpha=0.9)
    # nr.fit(X, y, animate=ANIMATE)
    # ridge_reg = ElasticNetRegression(l1_ratio=0.9, alpha=0.9, learning_rate=1e-1, loss=MSE)
    # ridge_reg.fit(X,y,epochs=2000, animate=ANIMATE)


    # lr = ElasticNetRegression(loss=MSE, alpha=0.9, learning_rate=LEARNING_RATE, scheduler_fn=scheduler, l1_ratio=0.9)
    # lr.fit(X, y, epochs=50, animate=True)


