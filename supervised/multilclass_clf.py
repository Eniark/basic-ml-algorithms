import numpy as np
from copy import deepcopy
from itertools import combinations


class OneVsRest:
    """This is a strategy where a multiclass classification problem is broken down into smaller binary classification problems.
        Creates n new classifiers, where n is the total number of classes.
            Each is trained using strategy "<current class> vs <other classes>" """
    def __init__(self, model):
        self.base_model = model
        self.classifiers = []


    def fit(self, X, y, epochs=1, animate=False):
        """Trains classifiers against other labels"""
        assert isinstance(y[0], (int, np.int8, np.int16, np.int32, np.int64)), 'Cannot use classifier as a regressor'
        num_of_models = len(np.unique(y))
        for i in range(num_of_models):
            new_y = np.where(y==i, 0, 1)
            new_model = deepcopy(self.base_model)
            self.classifiers.append(new_model)
            new_model.fit(X,new_y, epochs=epochs, animate=animate)

    def predict(self, X):
        """Performs prediction using highest score among the classifiers"""
        probabilities = 1 - np.array([model.predict(X, predict_proba=True) for model in self.classifiers])
        return np.argmax(probabilities)


class OneVsOne:
    """Creates n * (n-1)/2 classifiers. Each of them is trained against another class.
        Final prediction is made using mode of all predictions"""
    def __init__(self, model):
        self.base_model = model
        self.classifiers = []


    def fit(self, X, y, epochs=1, animate=False):
        """Create combinations of classes to train against each-other"""
        assert isinstance(y[0], (int, np.int8, np.int16, np.int32, np.int64)), 'Cannot use classifier as a regressor'
        combs = list(combinations(np.unique(y), r=2))
        for pair in combs:
            mask = np.logical_or(y==pair[0], y==pair[1])
            new_X = X[mask]
            new_y = y[mask]
            new_model = deepcopy(self.base_model)
            self.classifiers.append(new_model)
            new_model.fit(new_X,new_y, epochs=epochs, animate=animate)
    def predict(self, X):
        """Get the mode prediction as the result of all classifiers"""
        probabilities = np.array([model.predict(X, predict_proba=False) for model in self.classifiers]).flatten()
        uniq, cnt = np.unique(probabilities, return_counts=True)
        return uniq[np.argmax(cnt)]