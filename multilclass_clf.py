import numpy as np
from regressions import GradientDescentRegression






class OVR(GradientDescentRegression):
    def __init__(self, model):
        self.model = model


    def fit(self, X, y, epochs=1, animate=False):






