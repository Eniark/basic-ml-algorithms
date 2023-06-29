
from mllib.utilities.losses import BCE
from mllib.supervised.multilclass_clf import *
from mllib.utilities.lr_schedulers import *
from mllib.supervised.LinearRegression.regressions import Lasso, Ridge
from logistic_regression import LogisticRegression
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

plt.rcParams['axes.grid'] = True

if __name__=='__main__':
    N_FEATURES = 2
    ANIMATE = False
    LEARNING_RATE = 1e-1
    assert not ANIMATE or (ANIMATE and N_FEATURES==2), 'Can animate only 2 dimensions'

    scheduler = configure_power_scheduler(lr0=LEARNING_RATE, every=30)

    X,y = make_classification(n_features=N_FEATURES, n_classes=3, n_informative=2, n_redundant=0, n_repeated=0, n_clusters_per_class=1, random_state=1)

    lr = LogisticRegression(loss_fn=BCE, learning_rate=LEARNING_RATE, scheduler_fn=scheduler, regularization=None)
    ovr = OneVsOne(lr)
    ovr.fit(X,y, epochs=50, animate=ANIMATE)


    prd = [[1,1]]
    print(ovr.predict(prd))
    plt.show()
