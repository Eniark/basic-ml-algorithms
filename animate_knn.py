
import numpy as np
from sklearn.datasets import make_regression, make_classification
import matplotlib.pyplot as plt
from nearest_neighbors import *
np.random.seed(1)
plt.rcParams['axes.grid'] = True

if __name__=='__main__':
    # X, y = make_regression(n_features=2, noise=15)
    # y = np.round(y, 2)
    # X,y = make_classification(n_features=2, n_informative=1, n_redundant=0, n_repeated=0, n_clusters_per_class=1, random_state=1)

    X = np.array([[7,2], [5,4], [7,6], [2,3], [4,7], [8,1]])
    y = np.array([1,0,0,1,1,0])
    knn = KNeighboursClassifier(n_neighbors=3, algorithm='kd')
    knn.fit(X,y)
    X_pred = [0, 1]
    print(knn.predict(X_pred, show_graph=True))
