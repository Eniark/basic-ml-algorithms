
import numpy as np
from logistic_regression import LogisticRegression
from losses import BCE

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from utilities.lr_schedulers import *
np.random.seed(1)

plt.rcParams['axes.grid'] = True

if __name__=='__main__':
    N_FEATURES = 2
    ANIMATE = True
    LEARNING_RATE = 1e-1
    assert not ANIMATE or (ANIMATE and N_FEATURES==2), 'Can animate only 2 dimensions'

    scheduler = configure_power_scheduler(lr0=LEARNING_RATE, every=30)

    X,y = make_classification(n_features=N_FEATURES, n_informative=1, n_redundant=0, n_repeated=0, n_clusters_per_class=1, random_state=1)


    lr = LogisticRegression(loss_fn=BCE, learning_rate=LEARNING_RATE, scheduler_fn=scheduler)
    lr.fit(X,y, epochs=100, animate=ANIMATE)
    p = lr.predict([[0, -2], [2,1]])
    print(p)
