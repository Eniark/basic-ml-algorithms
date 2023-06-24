import numpy as np




# =========== Regression Losses ===========



class MSE:
    """Mean squared error. Usually used as a loss in training models.
        However, is strongly affected by the outliers because of the square in the calculations"""
    def calculate(self, y_true, y_pred, X):
        self.y_true = y_true
        self.y_pred = y_pred
        self.X = X
        self.value = np.mean(0.5 * (self.y_true - self.y_pred)**2)
        return self.value
    
    def get_derivative(self):
        derivative_theta = (-(self.y_true - self.y_pred).dot(self.X)) / self.X.shape[0]
        return derivative_theta


class MAE:
    """Mean absolute error. Is more robust to outliers"""
    def calculate(self, y_true, y_pred, X):
        self.y_true = y_true
        self.y_pred = y_pred
        self.X = X
        return np.abs(self.y_true - self.y_pred)

    def get_derivative(self):
        diff = self.y_true - self.y_pred 
        derivative_theta = np.zeros_like(self.y_true)        
        derivative_theta[diff<0] = -1/self.y_true
        derivative_theta[diff==0] = 0
        derivative_theta[diff>0] = 1/self.y_true
        return derivative_theta
    
class RMSE:
    """Root mean squared error"""
    def calculate(self, y_true, y_pred, X):
        self.y_true = y_true
        self.y_pred = y_pred
        self.X = X
        return np.sqrt(np.mean((self.y_true - self.y_pred)**2))

class MAPE:
    """Mean average percentage error"""
    def calculate(self, y_true, y_pred, X):
        self.y_true = y_true
        self.y_pred = y_pred
        self.X = X
        return np.mean(np.abs((self.y_true - self.y_pred)/self.y_true))







class R_squared:
    """Regression metric that shows how much variance of the data the line explains."""
    def calculate(self, y_true, y_pred, X):
        self.y_true = y_true
        self.y_pred = y_pred
        self.X = X

        def SST(y_true):
            return np.sum((y_true - np.mean(y_true))**2)
    
        def SSR(y_true, y_pred):
            return np.sum((y_pred - np.mean(y_true))**2)
    
        return 1 - SSR(self.y_true)/SST(self.y_true, self.y_pred)


class R_adjusted:
    """R-adjusted metric. With the increase of variables R-squared also increases.
        But if a new variable is not useful to the problem, R-squared will still increase.
         R-adjusted mitigates that effect"""
    def calculate(self, y_true, y_pred, X):
        self.y_true = y_true
        self.y_pred = y_pred
        self.X = X
        return 1 - (1 - R_squared(self.y_true, self.y_pred)**2)*(self.n - 1)/(self.n - self.p - 1)
# ===========================================



# =========== Classification Losses ===========
class BCE:
    """Binary Cross-Entropy Loss"""
    def calculate (self, y_true, y_pred, X):
        self.y_true = y_true
        self.y_pred = y_pred
        self.X = X
        self.value = -np.sum(y_true*np.log(y_pred) + (1 - y_true)*np.log(1 - y_pred)) / y_true.shape[0]
        return self.value

    def get_derivative(self):
        return self.X.T.dot(self.y_pred - self.y_true)







    


