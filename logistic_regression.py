import numpy as np
from regressions import GradientDescentRegression
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt



class LogisticRegression(GradientDescentRegression):
    def __init__(self, loss_fn, learning_rate, scheduler_fn=None, threshold=0.5):
        self.threshold = threshold
        super(LogisticRegression, self).__init__(loss=loss_fn, learning_rate=learning_rate, scheduler_fn=scheduler_fn)

    def sigmoid(self, logit):
        return np.where(logit>0, \
            (lambda: 1/(1 + np.exp(-logit)))(), \
            (lambda: np.exp(logit)/(1+np.exp(logit)))()
            )

        # return 1/(1 + np.exp(-logit))

    def fit(self, X, y, epochs=1, animate=False):
        X = self.standardize(X)
        super(LogisticRegression, self).fit(X, y, epochs=epochs, animate=animate)

    def _step(self, epoch, X, y):
        y_pred = self.sigmoid(np.dot(X, self.W))
        loss = self.loss_fn.calculate(y, y_pred, X) + self.regularization(self.W[1:])
        self._optimize()
        print(f'Epoch - {epoch} | Learning rate - {self.learning_rate} | Loss - {loss}')

        self.learning_rate = self.scheduler_fn(epoch)
        self.loss_history.append(loss)

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = self.sigmoid(X.dot(self.W))
        return np.where(y_pred <= self.threshold, 0, 1)

    def _animated_fit_step(self, epoch, X, y):
        plt.cla()
        x_line = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 10).reshape(-1, 1)
        m = -self.W[1]/self.W[-1]
        b = -self.W[0]/self.W[-1]
        y_line = x_line * m + b

        colors = ['red', 'blue']
        classes = np.unique(y)
        c_ = []
        for i in zip(colors, classes):
            c_.append(mpatches.Patch(color=i[0], label=f'Class {i[1]}'))
        plt.scatter(X[:, 1], X[:, 2], c=[colors[label] for label in y])

        plt.legend(handles=c_)
        plt.plot(x_line, y_line)
        self._step(epoch, X, y)
        plt.title(f'Epoch - {epoch} | Loss {self.loss_fn.__class__.__name__} {np.round(self.loss_fn.value,3)} | Learning rate - {self.learning_rate}');
