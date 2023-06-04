import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import combinations_with_replacement


class Regression:

    """Parent class for Regression"""

    def normalize(self, X):
        return (X - np.mean(X))/np.std(X)
    
    @staticmethod
    def get_line(x, m, b):
        return x * m + b


class OLSRegression(Regression):

    """Ordinary Least Squares regression"""

    def fit(self, X, y, animate=False):
        if X.ndim == 2:
            X = X.flatten()

        m = (np.mean(X) * np.mean(y) - np.mean(X*y))/(np.mean(X)**2 - np.mean(X**2))
        b = np.mean(y) - m * np.mean(X)
        if animate:
            x_line = range(int(np.min(X)), int(np.ceil(np.max(X))))
            y_line = super().get_line(x_line, m, b)

            plt.scatter(X, y)
            plt.plot(x_line, y_line);
            plt.show()
        return m, b


class GradientDescentRegression(Regression):

    """Regression with Gradient descent optimization variants"""

    def __init__(self, learning_rate=1e-2, loss=None):
        self.loss_fn = loss
        self.learning_rate = learning_rate
        self.loss_history = []


    def _initialize(self, n_parameters):
        limit = 1/np.sqrt(n_parameters)
        self.W = np.random.uniform(-limit, limit, size= (n_parameters, ))
    
    def _step(self, epoch, X, y):
        # print(X, self.W)
        y_pred = np.dot(X, self.W)
        # print(y_pred)
        loss = self.loss_fn.calculate(y, y_pred, X)
        print(f'Epoch {epoch}| loss {loss}')
        self.optimize()
        self.loss_history.append(loss)

    def optimize(self):
        derivative_theta = self.loss_fn.get_derivative()
        # print(self.W)
        self.W -= self.learning_rate * derivative_theta
        # print(self.W)

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.W)
        return y_pred

    def fit(self, X, y, epochs=1, animate=False):
        X = np.insert(X, 0, 1, axis=1)
        _, n_features = X.shape
        self._initialize(n_features)
        

        for epoch in range(1, epochs + 1):
            self._step(epoch, X, y)

class LinearRegression(GradientDescentRegression):

    """Linear Regression using gradient descent"""
    def __init__(self, learning_rate, loss):
        super(LinearRegression, self).__init__(loss=loss, learning_rate=learning_rate)
    

    def fit(self, X, y, epochs=1, animate=False):
        if animate:
            fig = plt.figure(figsize=(10,5))
            anim = FuncAnimation(fig, func=self.__animated_fit, fargs=(X,y), frames=epochs, interval=10, repeat=False)
            plt.show()

        else:
            super(LinearRegression, self).fit(X, y, epochs=1, animate=False) # call parents fit method which has no animation defined

    def __animated_fit(self, epoch, X, y):
        plt.cla()
        self._step(epoch, X, y)
        b, m = self.W
        x_line = range(int(np.min(X)), int(np.ceil(np.max(X))))
        y_line = super().get_line(x_line, m, b)

        plt.scatter(X[:, 1], y)
        plt.plot(x_line, y_line);


    
class SGDRegressor(GradientDescentRegression):
    def __init__(self, learning_rate, loss):
        super(SGDRegressor, self).__init__(loss=loss, learning_rate=learning_rate)
    
    def _step(self, epoch, X, y):
        idx = np.random.randint(X.shape[0])
        x_i = X[idx].reshape(1, 2)
        y_i = y[idx].reshape(1,)
        y_pred = np.dot(x_i, self.W).reshape(1,)
        loss = self.loss_fn.calculate(y_i, y_pred, x_i)
        print(f'Epoch {epoch}| loss {loss}')
        self.optimize()
        self.loss_history.append(loss)


    def fit(self, X, y, epochs=1, animate=False):

        X = np.insert(X, 0, 1, axis=1)
        _, n_features = X.shape
        self._initialize(n_features)

        if animate:
            assert n_features==2, 'There can only be one feature for visualization.' 
            fig = plt.figure(figsize=(10,5))
            anim = FuncAnimation(fig, func=self.__animated_fit, fargs=(X,y), frames=epochs, interval=10, repeat=False)
            plt.show()
            self.loss_history = self.loss_history[1:] # needed since FuncAnimation init_func and func parameters are the same 
                                                            # and one additional iteration occurs at the start
            plt.plot(range(1, epochs + 1), self.loss_history)
            plt.xlabel('Epoch')
            plt.ylabel(f'Loss - {self.loss_fn.__class__.__name__}')
            plt.show()
            return

       

        for epoch in range(epochs):
            self._step(epoch, X, y)

    def __animated_fit(self, epoch, X, y):
        plt.cla()
        self._step(epoch, X, y)
        b, m = self.W
        x_line = range(int(np.min(X)), int(np.ceil(np.max(X))))
        y_line = super().get_line(x_line, m, b)

        plt.scatter(X[:, 1], y)
        plt.plot(x_line, y_line);


class PolynomialRegression(GradientDescentRegression):
    def __init__(self, degree=2, learning_rate=0.01, loss=None):
        self.degree = degree
        super(PolynomialRegression, self).__init__(learning_rate, loss)


    def transform(self, X):
        # X_transformed = np.ones((X.shape[0], 1))
        
        # for deg in range(1, self.degree + 1):
        #     X_pow = np.power(X, deg)
        #     X_transformed = np.append(X_transformed, X_pow.reshape(-1, 1), axis=1)
        # return X_transformed

        n_samples, n_features = X.shape

        combs = [combinations_with_replacement(range(n_features), i) for i in range(self.degree + 1)]
        combs = [item for sublist in combs for item in sublist]
        n_output_features = len(combs)

        X_transformed = np.empty((n_samples, n_output_features))
        for idx, index_comb in enumerate(combs):
            X_transformed[:, idx] = np.prod(X[:, index_comb], axis=1)

        return X_transformed

    def fit(self, X, y, epochs=1, animate=False):
        X = self.transform(X)
        # X = self.normalize(X)
        _, n_features = X.shape
        self._initialize(n_features)
        if animate:
            fig = plt.figure(figsize=(10,5))
            anim = FuncAnimation(fig, func=self.__animated_fit, fargs=(X,y), frames=epochs, interval=10, repeat=False)
            plt.show()
            return

        for epoch in range(1, epochs + 1):
            self._step(epoch, X, y)

    def __animated_fit(self, epoch, X, y):
        plt.cla()
        self._step(epoch, X, y)
        x_line = range(int(np.min(X)), int(np.ceil(np.max(X[:, 1 ]))))
        y_line = np.dot(X, self.W)
        plt.scatter(X[:, 1], y)
        plt.plot(np.linspace(-2, int(np.ceil(np.max(X[:, 1 ]))), 101), y_line);