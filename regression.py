import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 

class Linear:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.n = len(X)
        self.X_mean = np.mean(X)
        self.Y_mean = np.mean(Y)
        self.calculate_coeff()
    
    def calculate_coeff(self):
        numerator = 0
        denominator = 0
        for x, y in zip(self.X, self.Y):
            numerator += (x - self.X_mean) * (y - self.Y_mean)
            denominator += (x - self.Y_mean) ** 2
        self.B1 = numerator / denominator
        self.B0 = self.Y_mean - (self.B1 * self.X_mean)

