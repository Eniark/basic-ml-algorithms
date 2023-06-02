
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

np.random.seed(1)
from matplotlib.animation import FuncAnimation


def get_line(x, m, b):
    return x * m + b




class Regression:

    def normalize(self):
        return (self.X - np.mean(X))/np.std(self.X)
    

    def _initialize(self, n_parameters):
        limit = 1/np.sqrt(n_parameters)
        self.W = np.random.uniform(-limit, limit, size= (n_parameters, ))



        

# class OLSRegression:
#     def __init__(self, X, y) -> None:
#         self.X = X
#         self.y = y

#     def fit(self):
#         m = (np.mean(self.X) * np.mean(self.y) - np.mean(self.X*self.y))/(np.mean(self.X)**2 - np.mean(self.X**2))
#         b = np.mean(self.y) - m * np.mean(self.X)
#         return m, b


class LinearRegression(Regression):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.super(LinearRegression, self).__init__()


    def loss_fn(self, y_true, y_pred):
        return np.mean(0.5*(y_true-y_pred)**2)
    
    def optimize(self, y_true, y_pred, X):
        derivative_theta = (-(y_true-y_pred).dot(X))/X.shape[0]
        self.W -= self.learning_rate* derivative_theta

   

    def fit(self, X, y, epochs=1, animate=False):
        if animate:
            fig = plt.figure(figsize=(10,5))
            anim = FuncAnimation(fig, func=self.animated_fit, fargs=(X,y), frames=epochs, interval=10)
            plt.show()
            return

        X = np.insert(X, 0, 1, axis=1)
        _, n_features = X.shape
        self._initialize(n_features)
        

        optimizer
        for i in range(epochs):
            optimizer.step()
            y_pred = np.dot(X, self.W)
            loss = self.loss_fn(y, y_pred)
            self.optimize(y, y_pred, X)
            print(f'Epoch {i}| loss {loss}')


    def animated_fit(self, i, X, y):

        plt.cla()
        X = np.insert(X, 0, 1, axis=1)
        _, n_features = X.shape
        if i==0:
            self._initialize(n_features)

        y_pred = np.dot(X, self.W)
        loss = self.loss_fn(y, y_pred)
        print(f'Epoch {i}| loss {loss}')
        self.optimize(y, y_pred, X)

        b, m = self.W
        x_line = range(np.min(X, axis=1), np.max(X,axis=1))
        y_line = get_line(x_line, m, b)

        plt.scatter(X[:, 1], y)
        plt.plot(x_line, y_line);


X, y = make_regression(n_features=1, noise=15)



# class SGDRegressor(Regression):
#     def __init__(self, learning_rate):
#         self.learning_rate = learning_rate


#     def _initialize(self, n_parameters):
#         limit = 1/np.sqrt(n_parameters)
#         self.W = np.zeros((2,))
#         # self.W = np.random.uniform(-limit, limit, size= (n_parameters, ))

#     def normalize(self, X):
#         return (X - np.mean(X))/np.std(X)


#     def loss_fn(self, y_true, y_pred):
#         return 0.5*(y_true-y_pred)**2
    
#     def optimize(self, y_true, y_pred, X):
#         derivative_W = -(y_true-y_pred)*X[1]
#         derivative_b = -(y_true-y_pred)

#         self.W[1] -= self.learning_rate* derivative_W
#         self.W[0] -= self.learning_rate* derivative_b

#     def fit(self, X, y, epochs=1, animate=False, normalize=False):
#         if normalize
#         if animate:
#             fig = plt.figure(figsize=(10,5))
#             anim = FuncAnimation(fig, func=self.animated_fit, fargs=(X,y), frames=500, interval=10)
#             plt.show()
#             return
#         X = np.insert(X, 0, 1, axis=1)
#         n_samples, n_features = X.shape
#         self._initialize(n_features)
#         idxs = np.arange(0, n_samples,1)

#         for epoch in range(epochs):
#             idx = np.random.choice(idxs) # choose random sample
#             x_i = X[idx]
#             y_i = y[idx]
#             y_pred = np.dot(x_i, self.W)
#             loss = self.loss_fn(y_i, y_pred)
#             self.optimize(y_i, y_pred, x_i)
#             print(f'Epoch {epoch}| loss {loss}')

#     def animated_fit(self, i, X, y):
#         plt.cla()
#         X = np.insert(X, 0, 1, axis=1)
#         n_samples, n_features = X.shape
#         self._initialize(n_features)
#         idxs = np.arange(0, n_samples,1)

#         idx = np.random.choice(idxs)
#         x_i = X[idx]
#         y_i = y[idx]
#         y_pred = np.dot(x_i, self.W)
#         loss = self.loss_fn(y_i, y_pred)
#         self.optimize(y_i, y_pred, x_i)
#         print(f'Epoch {i}| loss {loss}')

#         plt.scatter(X[:, 1], y)
#         plt.plot([self.W[0] * i + self.W[1] for i in range(-5, n_samples+1)])


linear_reg = LinearRegression(learning_rate=1e-1)
# linear_reg.fit(X,y,epochs=5)
