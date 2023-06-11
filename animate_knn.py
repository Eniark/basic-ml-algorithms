
import numpy as np
from sklearn.datasets import make_regression, make_classification
import matplotlib.pyplot as plt
from nearest_neighbors import *
np.random.seed(1)
plt.rcParams['axes.grid'] = True

if __name__=='__main__':
    # X, y = make_regression(n_features=2, noise=15)
    # y = np.round(y, 2)
    X,y = make_classification(n_features=2, n_informative=1, n_redundant=0, n_repeated=0, n_clusters_per_class=1, random_state=1)
    # knn = KNeighboursRegressor(n_neighbors=5)
    # knn.fit(X,y)
    # X_pred = [2, 0.4]
    # print(knn.predict(X_pred, show_graph=True))


    kdtree = KDTree(X,y)
    kdtree.build(X)