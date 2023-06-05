import numpy as np




# =========== Regression Losses ===========



class MSE:
    def calculate(self, y_true, y_pred, X):
        self.y_true = y_true
        self.y_pred = y_pred
        self.X = X
        # print('Loss FN:')
        # print(self.y_true - self.y_pred)
        # print(np.mean(0.5 * (self.y_true - self.y_pred)**2))
        return np.mean(0.5 * (self.y_true - self.y_pred)**2)
    
    def get_derivative(self):
        derivative_theta = (-(self.y_true - self.y_pred).dot(self.X)) / self.X.shape[0]
        return derivative_theta


class MAE:
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
        print(derivative_theta)
        return derivative_theta
    
class RMSE:
    def calculate(self, y_true, y_pred, X):
        self.y_true = y_true
        self.y_pred = y_pred
        self.X = X
        return np.sqrt(np.mean((self.y_true - self.y_pred)**2))

class MAPE:
    def calculate(self, y_true, y_pred, X):
        self.y_true = y_true
        self.y_pred = y_pred
        self.X = X
        return np.mean(np.abs((self.y_true - self.y_pred)/self.y_true))







class R_squared:
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
    def calculate(self, y_true, y_pred, X):
        self.y_true = y_true
        self.y_pred = y_pred
        self.X = X
        return 1 - (1 - R_squared(self.y_true, self.y_pred)**2)*(self.n - 1)/(self.n - self.p - 1)
# ===========================================



# =========== Classification Losses ===========





    


