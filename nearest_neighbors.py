
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

def projection(p,v):
    return v.dot(v.T)/v.T.dot(v) * p

def euclidean_distance(x1, x2):
    distance = 0
    for coord in range(len(x1)):
        distance += (x1[coord]-x2[coord])**2
    return np.round(np.sqrt(distance), 3)

class KNN:
    """Parent Class for K-Nearest Neighbors"""
    def __init__(self, n_neighbors=5, algorithm='brute'):
        assert n_neighbors%2!=0, 'Please specify an odd number of neighbors'
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm

    # def euclidean_distance(self, x1, x2):
    #     distance = 0
    #     for coord in range(len(x1)):
    #         distance += (x1[coord]-x2[coord])**2
    #     return distance

    def fit(self, X, y, animate=False):
        """Fit according to chosen algoirthm. In case of brute algorithm selected -> no fitting is needed."""
        self.X = X
        self.y = y
        if self.algorithm=='kd':
            # create kd-tree
            self.kdtree = KDTree(X,y, k=self.n_neighbors)

        elif self.algorithm=='bt':
            # create ball-tree
            pass

    def predict(self, X_pred, show_graph=False):
        """Predict according to algorithm selected"""
        if self.algorithm=='brute':
            distances = [euclidean_distance(node, X_pred) for node in self.X]
            idxs = np.argsort(distances)[:self.n_neighbors]
        elif self.algorithm=='kd':
            assert self.kdtree.root, 'Please fit the tree first.'
            idxs = self.kdtree.nearest_neighbors(X_pred)[:self.n_neighbors]
        
        prediction = self._predict(idxs)
        if show_graph:
            self.graph(idxs, X_pred)
        return prediction

    def graph(self, idxs, X_pred):
        """Display points on graph"""
        mask = np.ones(self.X.shape[0], dtype=bool)
        mask[idxs] = False
        X_masked = self.X[mask]
        plt.scatter(X_masked[:,0], X_masked[:,1], label='Other points', edgecolors='b')

        plt.scatter(X_pred[0], X_pred[1], c='red', label='Input point', edgecolors='b')
        plt.title(f'{self.__class__.__name__}')

class KNeighboursClassifier(KNN):
    """KNN class for classification tasks."""
    def __init__(self, n_neighbors=5, algorithm='brute'):
        super(KNeighboursClassifier, self).__init__(n_neighbors, algorithm)

    def graph(self, idxs, X_pred):
        """Invoke parents method and plot points coloured by class"""
        super(KNeighboursClassifier, self).graph(idxs, X_pred)
        plt.scatter(self.X[idxs, 0], self.X[idxs, 1], c=self.y[idxs], label='Neighbors')
        plt.legend()
        plt.show()
        

    def _predict(self, closest_idxs):
        """Predict method: mathematical mode of neighbors' classes"""
        closest_classes = self.y[closest_idxs]
        unique, counts = np.unique(closest_classes, return_counts=True)
        max_count = 0
        for idx, count in enumerate(counts):
            if count>max_count:
                max_count = count
                class_ = unique[idx]

        return class_


class KNeighboursRegressor(KNN):
    """KNN class for regression tasks."""

    def __init__(self, n_neighbors=5, algorithm='brute'):
        super(KNeighboursRegressor, self).__init__(n_neighbors, algorithm)

    def graph(self, idxs, X_pred):
        """Plot points and add text to neighbors to show target values of each of them"""
        for i in idxs:
            plt.text(self.X[i,0], self.X[i,1], self.y[i])
        
        super(KNeighboursRegressor, self).graph(idxs, X_pred)
        plt.scatter(self.X[idxs, 0], self.X[idxs, 1], c='green', label='Neighbors')
        plt.legend()
        plt.show()
    

    def _predict(self, closest_idxs):
        """Predict method: average of neighbors"""
        value = np.mean(self.y[closest_idxs])
        return value
  



class Node:
    """Node class to store information in a tree structure"""
    def __init__(self, data, y, axis, idx_in_ds):
        self.X = data
        self.y = y
        self.idx = idx_in_ds
        self.left = None
        self.right = None
        self.axis = axis
        self.__info_string = f'{self.__class__.__name__}{self.X} || Idx: {self.idx}'
        
    def __str__(self):
        return self.__info_string
    
    def __repr__(self):
        return self.__info_string
class KDTree:
    """KD-Tree implementation. It's a k-dimensional binary tree to split the data and minimize search time for neighbors."""
    def __init__(self, X, y, k=1):
        self.X = X
        self.y = y
        self.neighbors = []
        self.k = k

        idxs = np.arange(y.shape[0])
        self.root = self.build(X, y, idxs_in_dataset=idxs)


    def build(self, X, y, idxs_in_dataset=None, depth=0):
        """Recursively build the tree by splitting by median of each dimension on each call"""
        k = X.shape[1]
        axis = depth % k
        if X.shape[0]==0:
            return
        idxs = X[:, axis].argsort()
        sorted_X = X[idxs]
        sorted_idxs = idxs_in_dataset[idxs]
        median_idx = sorted_X.shape[0] // 2
        
        node = Node(data=sorted_X[median_idx, :], y=y[median_idx], axis=axis, idx_in_ds=sorted_idxs[median_idx])
        node.left = self.build(X=sorted_X[:median_idx, :], y=y[:median_idx], idxs_in_dataset=sorted_idxs[:median_idx], depth=depth+1)
        node.right = self.build(X=sorted_X[median_idx + 1:, :], y=y[median_idx + 1:], idxs_in_dataset=sorted_idxs[median_idx + 1:], depth=depth+1)
        
        return node
 
    # Invoked by nearest_neighbors method
    def _nearest_neighbors(self, X, current_node, best_node=None, best_distance=np.inf, neighbors=[]):
        """Traverse the tree and search for k nearest neighbors. Improves search by pruning other halves of tree"""
        if not current_node:
            return neighbors, best_node, best_distance
        d = euclidean_distance(current_node.X, X)

        if d < best_distance:
            best_distance = d
            best_node = current_node

        if len(neighbors) < self.k:
            neighbors.append((current_node, d))
        elif len(neighbors) >= self.k and d < neighbors[-1][-1]:
            neighbors = sorted(neighbors, key=lambda x: x[1])
            neighbors[-1] = (current_node, d)
        if X[current_node.axis] < current_node.X[current_node.axis]:
            good_side = current_node.left
            bad_side = current_node.right
        else:
            good_side = current_node.right
            bad_side = current_node.left

        neighbors, best_node, best_distance = self._nearest_neighbors(X, good_side, best_node, best_distance, neighbors=neighbors)
        if abs(current_node.X[current_node.axis] - X[current_node.axis]) < best_distance:
            neighbors, best_node, best_distance = self._nearest_neighbors(X, bad_side, best_node, best_distance, neighbors=neighbors)


        return neighbors, best_node, best_distance
    

    def nearest_neighbors(self, X):
        """Search k closest neighbors to query point X"""
        neighbors, _, _ = self._nearest_neighbors(X, self.root)
        neighbors = sorted(neighbors, key=lambda x: x[-1])
        idxs = np.array(list(map(lambda x: x[0].idx, neighbors)))
        return idxs



        




        

