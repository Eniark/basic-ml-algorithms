import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import combinations_with_replacement
plt.rcParams['axes.grid'] = True





class Regression:

    """Parent class for Regression"""

    def standardize(self, X):
        # if X.ndim==2:
            # return (X - np.mean(X))/np.std(X)
        return (X - np.mean(X, axis=0))/np.std(X, axis=0)
    
    def normalize(self, X):
        min_x = np.min(X)
        return (X - min_x)/(np.max(X) - min_x)
    
    @staticmethod
    def get_line(x, m, b):
        return x * m + b


class OLSRegression(Regression):

    """Ordinary Least Squares regression"""

    def fit(self, X, y, animate=False):

        X = X.flatten()
        m = (np.mean(X) * np.mean(y) - np.mean(X*y))/(np.mean(X)**2 - np.mean(X**2))
        b = np.mean(y) - m * np.mean(X)
        if animate:
            # x_line = range(int(np.min(X)), int(np.ceil(np.max(X))))
            # y_line = super().get_line(x_line, m, b)
            x_line = np.linspace(np.min(X), np.max(X), X.shape[0]).reshape(-1, 1)
            x_line = np.insert(x_line, 0, 1, axis=1)
            y_line = np.dot(x_line, np.array([b, m]))
            plt.scatter(X, y)
            plt.plot(x_line[:, 1], y_line)
            plt.title('Ordinary Least Squares');
            plt.show()
        return m, b


class GradientDescentRegression(Regression):

    """Regression with Gradient descent optimization variants"""

    def __init__(self, loss, learning_rate=1e-2):
        self.regularization = lambda w: 0
        self.regularization.derivative = lambda w: 0

        self.loss_fn = loss
        self.learning_rate = learning_rate
        self.loss_history = []


    def _initialize(self, n_parameters):
        limit = 1/np.sqrt(n_parameters)
        self.W = np.random.uniform(-limit, limit, size=(n_parameters, ))
        print(self.W)
    
    def _step(self, epoch, X, y):
        y_pred = np.dot(X, self.W)
        loss = self.loss_fn.calculate(y, y_pred, X) + self.regularization(self.W[1:])
        self._optimize()
        self.loss_history.append(loss)

    def _optimize(self):
        derivative_delta = self.loss_fn.get_derivative() + self.regularization.derivative(self.W[1:])
        self.W -= self.learning_rate * derivative_delta

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.W)
        return y_pred

    def fit(self, X, y, epochs=1, animate=False):
        X = np.insert(X, 0, 1, axis=1)
        _, n_features = X.shape
        self._initialize(n_features)
        if animate:    
            self._animated_fit(X,y, epochs + 1)
        else:
            for epoch in range(1, epochs + 1):
                self._step(epoch, X, y)


    def _animated_fit(self, X, y, epochs):
        fig = plt.figure(figsize=(10,5))
        anim = FuncAnimation(fig, func=self._animated_fit_step, init_func=lambda: 0, fargs=(X,y), frames=epochs, interval=10, repeat=False)
        plt.show()
        plt.plot(range(1, epochs + 1), self.loss_history)
        plt.xlabel('Epoch')
        plt.title(f'Loss - {self.loss_fn.__class__.__name__}')
        plt.ylabel('Error')
        plt.show()


    def _animated_fit_step(self, epoch, X, y):
        plt.cla()
        x_line = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), X.shape[0]).reshape(-1, 1)
        x_line = np.insert(x_line, 0, 1, axis=1)
        y_line = np.dot(x_line, self.W)

        plt.scatter(X[:, 1], y)
        plt.plot(x_line[:, 1], y_line)
        self._step(epoch, X, y)
        plt.title(f'Epoch {epoch} | {self.loss_fn.__class__.__name__} loss {np.round(self.loss_fn.value,3)}');

class LinearRegression(GradientDescentRegression):

    """Linear Regression using Gradient Descent"""

    def __init__(self, learning_rate, loss):
        super(LinearRegression, self).__init__(loss=loss, learning_rate=learning_rate)
    

    def fit(self, X, y, epochs=1, animate=False):
        super(LinearRegression, self).fit(X, y, epochs=epochs, animate=animate)


    
class SGDRegressor(GradientDescentRegression):

    """Linear Regression using Stochastic Gradient Descent"""
    def __init__(self, learning_rate, loss):
        super(SGDRegressor, self).__init__(loss=loss, learning_rate=learning_rate)
    
    def fit(self, X, y, epochs=1, animate=False):
        super(SGDRegressor, self).fit(X,y, epochs, animate)

    def _step(self, epoch, X, y):
        idx = np.random.randint(X.shape[0])
        x_i = X[idx].reshape(1, 2)
        y_i = y[idx].reshape(1,)
        super(SGDRegressor, self)._step(epoch, x_i, y_i)



class PolynomialRegression(GradientDescentRegression):
    """Multiple regression using Gradient Descent algorithm"""

    """Suggestion: use learning_rate<=0.01. Very unstable. Skipping this for now."""

    def __init__(self, loss, degree=2, learning_rate=0.01 ):
        self.degree = degree
        super(PolynomialRegression, self).__init__(loss=loss, learning_rate=learning_rate)


    def transform(self, X):
        n_samples, n_features = X.shape
        combinations_of_features = [combinations_with_replacement(range(n_features), i) for i in range(self.degree + 1)]
        combinations_of_features = [item for sublist in combinations_of_features for item in sublist]
        n_resulting_features = len(combinations_of_features)
        X_transformed = np.empty((n_samples, n_resulting_features))
        for idx, index_combinations in enumerate(combinations_of_features):
            X_transformed[:, idx ] = np.prod(X[:, index_combinations], axis=1)

        return X_transformed[:, 1:]

    def fit(self, X, y, epochs=1, animate=False):
    #     assert 0 < self.degree <= 2, 'Degrees > 2 currently not supported.'
        X = self.transform(X)
        # X = self.normalize(X)
        if animate:
            X = np.insert(X, 0, 1, axis=1)
            _, n_features = X.shape
            self._initialize(n_features)
            self._animated_fit(X,y,epochs)

        else:
            for epoch in range(1, epochs + 1):
                self._step(epoch, X, y)

    def _animated_fit_step(self, epoch, X, y):
        plt.cla()

        x_line = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), X.shape[0]).reshape(-1, 1)
        x_line_tr = self.transform(x_line)
        # x_line_tr = self.normalize(x_line_tr)
        x_line_tr = np.insert(x_line_tr, 0, 1, axis=1)
                
        y_line = np.dot(x_line_tr, self.W)
        plt.scatter(X[:, 1], y)
        plt.plot(x_line, y_line);

        self._step(epoch, X, y)



class Regularization:
    def __init__(self, alpha):
        self.alpha = alpha

class Lasso(Regularization):    
    def __init__(self, alpha):
        super(Lasso, self).__init__(alpha=alpha)
    
    def __call__(self, W):
        return self.alpha * np.linalg.norm(W) ###

    def derivative(self, W):
        return self.alpha * np.sum(np.sign(W))
    
class Ridge(Regularization):
    def __init__(self, alpha):
        super(Ridge, self).__init__(alpha=alpha)

    def __call__(self, W):
        return self.alpha * 0.5 * W.T.dot(W)

    def derivative(self, W):
        return self.alpha * W
    

class ElasticNet(Regularization):
    def __init__(self, alpha, l1_ratio=0.5):
        self.l1_ratio = l1_ratio
        super().__init__(alpha)

    def __call__(self, W):
        return self.l1_ratio * self.alpha * np.linalg.norm(W) + (1 - self.l1_ratio) * 0.5 * self.alpha * W.T.dot(W)
    
    def derivative(self, W):
        return (self.l1_ratio * np.sign(W) + (1-self.l1_ratio) * W) *  self.alpha

class LassoRegression(GradientDescentRegression):
    def __init__(self, loss, alpha=0.5, learning_rate=0.01):
        super(LassoRegression, self).__init__(loss, learning_rate)
        self.regularization = Lasso(alpha=alpha)


    def fit(self, X, y, epochs=1, animate=False):
        X = self.standardize(X)
        return super().fit(X, y, epochs, animate)

