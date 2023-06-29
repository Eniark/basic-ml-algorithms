import numpy as np
class Regularization:
    """Parent class for regularization"""

    def __init__(self, alpha):
        self.alpha = alpha


class Lasso(Regularization):
    """Regularization which when given high alpha tends to make weights of unimportant features equal zero"""

    def __init__(self, alpha):
        super(Lasso, self).__init__(alpha=alpha)

    def __call__(self, W):
        return self.alpha * np.linalg.norm(W)

    def derivative(self, W):
        return self.alpha * np.sign(W)  # use subgradient


class Ridge(Regularization):
    """Regulariation technique used when dataset contains multicolinearity: features that are explained by other features"""

    def __init__(self, alpha):
        super(Ridge, self).__init__(alpha=alpha)

    def __call__(self, W):
        return self.alpha * 0.5 * W.T.dot(W)

    def derivative(self, W):
        return self.alpha * W


class ElasticNet(Regularization):
    """Regularization that combines Lasso and Ridge. """

    """
        | ====== Parameters ====== |
        l1_ratio: defines the influence of Lasso in the calculations. 
            Higher l1_ratio -> will be more like Lasso regression and vice-versa.
    """

    def __init__(self, alpha, l1_ratio=0.5):
        self.l1_ratio = l1_ratio
        super().__init__(alpha)

    def __call__(self, W):
        return self.l1_ratio * self.alpha * np.linalg.norm(W) + (1 - self.l1_ratio) * 0.5 * self.alpha * W.T.dot(W)

    def derivative(self, W):
        return (self.l1_ratio * np.sign(W) + (1 - self.l1_ratio) * W) * self.alpha
