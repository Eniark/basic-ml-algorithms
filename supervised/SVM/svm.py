import numpy as np
from mllib.supervised.LinearRegression.regressions import GradientDescent
import matplotlib.pyplot as plt

class SVM(GradientDescent):
    """Currently this is SVC"""
    def __init__(self, loss, learning_rate, scheduler_fn=None, C=1, kernel='linear'):
        super(SVM, self).__init__(loss=loss, learning_rate=learning_rate, scheduler_fn=scheduler_fn)
        self.C = C
        self.kernel = kernel
    def fit(self, X, y, epochs=1, animate=False):
        classes = np.unique(y)
        assert len(classes)==2, 'SVM is used only as a binary classifier.'
        X = self.standardize(X)
        self.conversion = dict([(-1, np.min(classes)), (1, np.max(classes))])
        y[y == np.min(classes)] = -1
        y[y == np.max(classes)] = 1
        # Classes should be -1 and +1
        # Update parameters depending on t >1 or <=1
        # Compute gradient of regularization term aswell.

        super(SVM, self).fit(X=X,y=y,epochs=epochs,animate=animate)

    def _step(self, epoch, X, y):
        regularization_term = self.W.dot(self.W) * 0.5
        margins = (y * np.dot(X, self.W))
        hinge_loss = self.loss_fn.calculate(y, margins, X)
        loss = hinge_loss + self.C * regularization_term

        self._optimize(y, margins, X)
        print(f'Epoch - {epoch} | Learning rate - {self.learning_rate} | Loss - {loss}')
        self.learning_rate = self.scheduler_fn(epoch)
        self.loss_history.append(loss)


    def _optimize(self, y_true, margins, X):
        dW = np.zeros(self.W.shape[0])

        dW[1:] = np.dot((margins < 1) * y_true, X[:, 1:])
        dW[1:] -= self.C * self.W[1:]
        dW[0] = np.sum((margins < 1) * y_true)
        print(dW)
        # derivative_delta = self.loss_fn.get_derivative(y_true, margins, X) # + regularization_derivative
        self.W -= self.learning_rate * dW
        # self.weights -= self.learning_rate * dW
        # self.bias -= self.learning_rate * db