
import numpy as np

import matplotlib.pyplot as plt

class KNN:
    def __init__(self, n_neighbors=5, algorithm='brute'):
        assert n_neighbors%2!=0, 'Please specify an odd number of neighbors'
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm

    def euclidean_distance(self, x1, x2):
        distance = 0
        for coord in range(len(x1)):
            distance += (x1[coord]-x2[coord])**2
        return distance

    def predict(self, X_pred, show_graph=False):
        distances = [self.euclidean_distance(node, X_pred) for node in self.X]
        idxs = np.argsort(distances)[:self.n_neighbors]
        prediction = self._predict(idxs)
        if show_graph:
            self.graph(idxs, X_pred)
        return prediction

    def graph(self, idxs, X_pred):
        mask = np.ones(self.X.shape[0], dtype=bool)
        mask[idxs] = False
        X_masked = self.X[mask]
        plt.scatter(X_masked[:,0], X_masked[:,1], label='Other points')

        plt.scatter(X_pred[0], X_pred[1], c='black', label='Input point')
        plt.title(f'{self.__class__.__name__}')

class KNeighboursClassifier(KNN):
    def __init__(self, n_neighbors=5, algorithm='brute'):
        super(KNeighboursClassifier, self).__init__(n_neighbors, algorithm)

    def graph(self, idxs, X_pred):
        super(KNeighboursClassifier, self).graph(idxs, X_pred)
        plt.scatter(self.X[idxs, 0], self.X[idxs, 1], c=self.y[idxs])
        plt.legend()
        plt.show()
        

    def fit(self, X, y, animate=False):
        if self.algorithm=='brute':
            self.X = X
            self.y = y
        elif self.algorithm=='kd':
            # fit kd-tree
            pass
        elif self.algorithm=='bt':
            # fit ball-tree
            pass

    def _predict(self, closest_idxs):
        closest_points = self.y[closest_idxs]
        unique, counts = np.unique(closest_points, return_counts=True)
        max_count = 0
        for idx, count in enumerate(counts):
            if count>max_count:
                max_count = count
                class_ = unique[idx]

        return class_


class KNeighboursRegressor(KNN):
    def __init__(self, n_neighbors=5, algorithm='brute'):
        super(KNeighboursRegressor, self).__init__(n_neighbors, algorithm)

    def graph(self, idxs, X_pred):
        for i in idxs:
            plt.text(self.X[i,0], self.X[i,1], self.y[i])
        
        super(KNeighboursRegressor, self).graph(idxs, X_pred)
        plt.scatter(self.X[idxs, 0], self.X[idxs, 1], c='green', label='Neighbors')
        plt.legend()
        plt.show()
       

    def fit(self, X, y, animate=False):
        if self.algorithm=='brute':
            self.X = X
            self.y = y
        elif self.algorithm=='kd':
            # fit kd-tree
            pass
        elif self.algorithm=='bt':
            # fit ball-tree
            pass

    def _predict(self, closest_idxs):
        value = np.mean(self.y[closest_idxs])
        return value
  



class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

class KDTree:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.root = self.build(X)
        

    def build(self, X, depth=0):
        k = X.shape[1]
        axis = depth % k

        if X.shape[0]==0:
            return
        sorted_data = np.sort(X, axis=axis)
        median_idx = X.shape[0] // 2

        node = Node(sorted_data[median_idx, axis])

        node.left = self.build(X=sorted_data[:median_idx, :], depth=depth+1)
        node.right = self.build(X=sorted_data[median_idx + 1:, :], depth=depth+1)

        return node
 