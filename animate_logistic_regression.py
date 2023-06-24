
import numpy as np
from logistic_regression import LogisticRegression
from losses import BCE
from multilclass_clf import *
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from regressions import Lasso, Ridge
from utilities.lr_schedulers import *
np.random.seed(1)

plt.rcParams['axes.grid'] = True

if __name__=='__main__':
    N_FEATURES = 2
    ANIMATE = False
    LEARNING_RATE = 1e-1
    assert not ANIMATE or (ANIMATE and N_FEATURES==2), 'Can animate only 2 dimensions'

    scheduler = configure_power_scheduler(lr0=LEARNING_RATE, every=30)

    X,y = make_classification(n_features=N_FEATURES, n_classes=3, n_informative=2, n_redundant=0, n_repeated=0, n_clusters_per_class=1, random_state=1)
    s = plt.scatter(X[:,0], X[:,1], c=y)
    plt.legend(
        handles=s.legend_elements()[0],
        labels=map(lambda label: f'Class {label}', np.unique(y))
    )
    plt.show()
    # lasso = Ridge(alpha=0.1)
    lr = LogisticRegression(loss_fn=BCE, learning_rate=LEARNING_RATE, scheduler_fn=scheduler, regularization=None)
    ovr = OneVsOne(lr)
    ovr.fit(X,y, epochs=50, animate=ANIMATE)
    print(ovr.predict([[-1,-1]]))

