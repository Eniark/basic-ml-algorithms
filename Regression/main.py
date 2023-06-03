import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



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


    def __initialize(self, n_parameters):
        limit = 1/np.sqrt(n_parameters)
        self.W = np.random.uniform(-limit, limit, size= (n_parameters, ))
    
    def __step(self, epoch, X, y):
        y_pred = np.dot(X, self.W)
        loss = self.loss_fn(y, y_pred)
        print(f'Epoch {epoch}| loss {loss}')
        self.optimize(y, y_pred, X)

    def fit(self, X, y, epochs=1, animate=False):
        X = np.insert(X, 0, 1, axis=1)
        _, n_features = X.shape
        self.__initialize(n_features)
       

        for epoch in range(epochs):
            super().__step(epoch, X, y)
    

class LinearRegression(GradientDescentRegression):

    """Linear Regression using gradient descent"""
    def __init__(self, learning_rate, loss):
        super(LinearRegression, self).__init__(loss_fn=loss, learning_rate=learning_rate)
    
    def optimize(self, y_true, y_pred, X):
        derivative_theta = (-(y_true-y_pred).dot(X))/X.shape[0]
        self.W -= self.learning_rate * derivative_theta
    
    def fit(self, X, y, epochs=1, animate=False):
        pass

    def __animated_fit(self, epoch, X, y):
        plt.cla()
        super().__step(epoch, X, y)
        b, m = self.W
        x_line = range(int(np.min(X)), int(np.ceil(np.max(X))))
        y_line = super().get_line(x_line, m, b)

        plt.scatter(X[:, 1], y)
        plt.plot(x_line, y_line);


class SGDRegressor(GradientDescentRegression):
    def __init__(self, learning_rate, loss):
        super(SGDRegressor, self).__init__(loss=loss, learning_rate=learning_rate)
    
    def __step(self, epoch, X, y):
        x_i = np.random.choice(X) 
        y_pred = np.dot(x_i, self.W)
        loss = self.loss_fn(y, y_pred)
        print(f'Epoch {epoch}| loss {loss}')
        self.optimize(y, y_pred, x_i)


    def fit(self, X, y, animate=False):
        if animate:
            assert n_features==2, 'There can only be one feature for visualization.' 
            fig = plt.figure(figsize=(10,5))
            anim = FuncAnimation(fig, func=self.__animated_fit, fargs=(X,y), frames=epochs, interval=10, repeat=False)
            plt.show()
            return

        else:
            super().fit(X,y)
    def __animated_fit(self, epoch, X, y):
        plt.cla()
        self.__step(epoch, X, y)
        b, m = self.W
        x_line = range(int(np.min(X)), int(np.ceil(np.max(X))))
        y_line = super().get_line(x_line, m, b)

        plt.scatter(X[:, 1], y)
        plt.plot(x_line, y_line);
