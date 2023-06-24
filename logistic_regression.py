import numpy as np
from regressions import GradientDescentRegression
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt



class LogisticRegression(GradientDescentRegression):
    """A binary classification model. Uses binary-cross entropy loss to train
        as MSE is non-convex(contains many local minima and maxima) for this model."""
    def __init__(self, loss_fn, learning_rate, scheduler_fn=None, regularization=None, threshold=0.5):
        self.threshold = threshold
        super(LogisticRegression, self).__init__(loss=loss_fn,
                                                 learning_rate=learning_rate,
                                                 scheduler_fn=scheduler_fn,
                                                 regularization=regularization)

    def sigmoid(self, logit):
        """Converts the calculations into numbers that can be interpreted as probabilities."""
        return np.where(logit>0, \
            (lambda: 1/(1 + np.exp(-logit)))(), \
            (lambda: np.exp(logit)/(1+np.exp(logit)))()
            )

        # return 1/(1 + np.exp(-logit))

    def fit(self, X, y, epochs=1, animate=False):
        classes = np.unique(y)
        assert len(classes)==2, 'Logistic Regression is used only as a binary classifier.'
        X = self.standardize(X)
        # y[y==np.min(classes)] = 0
        # y[y==np.max(classes)] = 1
        super(LogisticRegression, self).fit(X, y, epochs=epochs, animate=animate)

    def _step(self, epoch, X, y):
        """Overloads the _step and performs calculation and optimization"""
        y_pred = self.sigmoid(np.dot(X, self.W))
        loss = self.loss_fn.calculate(y, y_pred, X) + self.regularization(self.W[1:])
        self._optimize()
        print(f'Epoch - {epoch} | Learning rate - {self.learning_rate} | Loss - {loss}')

        self.learning_rate = self.scheduler_fn(epoch)
        self.loss_history.append(loss)

    def predict(self, X, predict_proba=False):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = self.sigmoid(X.dot(self.W))
        return y_pred if predict_proba else np.where(y_pred <= self.threshold, 0, 1)

    def _animated_fit_step(self, epoch, X, y):
        plt.cla()
        x_line = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 10).reshape(-1, 1)
        m = -self.W[1]/self.W[-1]
        b = -self.W[0]/self.W[-1]
        y_line = x_line * m + b
        colors = ['red', 'blue']
        classes = np.unique(y)
        # y[y==np.min(classes)] = 0
        # y[y==np.max(classes)] = 1
        c_ = []
        for i in zip(colors, classes):
            c_.append(mpatches.Patch(color=i[0], label=f'Class {i[1]}'))
        plt.scatter(X[:, 1], X[:, 2], c=[colors[label] for label in y]) # what if I receive classes [1,3]?

        plt.legend(handles=c_)
        plt.plot(x_line, y_line)
        self._step(epoch, X, y)
        plt.title(f'Epoch - {epoch} | Loss {self.loss_fn.__class__.__name__} {np.round(self.loss_fn.value,3)} | Learning rate - {self.learning_rate}');

    def __repr__(self):
        return f"{self.__class__.__name__} model"