import numpy as np




# =========== Regression Losses ===========
def MSE(y_true, y_pred):
    return np.mean(0.5 * (y_true - y_pred)**2)

def MAE(y_true, y_pred):
    return np.abs(y_true - y_pred)

def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)/y_true))


def R_squared(y_true, y_pred):
    def SST(y_true):
        return np.sum((y_true - np.mean(y_true))**2)
    
    def SSR(y_true, y_pred):
        return np.sum((y_pred - np.mean(y_true))**2)
    

    return 1 - SSR(y_true)/SST(y_true, y_pred)
# ===========================================



# =========== Classification Losses ===========





    


