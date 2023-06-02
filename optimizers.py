



class SCG:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate


    def step(self, y_true, y_pred, x_i):
        derivative_W = -(y_true-y_pred)*x_i[1]
        derivative_b = -(y_true-y_pred)

        self.W[1] -= self.learning_rate* derivative_W
        self.W[0] -= self.learning_rate* derivative_b

