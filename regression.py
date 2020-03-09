import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from math import sqrt
from sklearn.datasets import make_regression

class Linear:
    def __init__(self, X, Y, lr=0.005, num_iter=1000):
        self.X = X
        self.Y = Y
        self.n = len(X)
        self.lr = lr
        self.num_iter = num_iter
        self.theta0 = np.random.rand()
        self.theta1 = np.random.rand()
    
    def hypothesis(self, x):
        return self.theta0 + (self.theta1 * x)

    def calculate_coeff(self):
        numerator = 0
        denominator = 0
        for x, y in zip(self.X, self.Y):
            numerator += (x - self.X_mean) * (y - self.Y_mean)
            denominator += (x - self.Y_mean) ** 2
        self.theta1 = numerator / denominator
        self.theta0 = self.Y_mean - (self.theta1 * self.X_mean)
    
    def rmse_metric(self, actual, predicted):
        sum_error = 0.0
        for y, y_hat in zip(actual, predicted):
            prediction_error = y_hat - y
            sum_error += (prediction_error ** 2)
        mean_error = sum_error / float(len(actual))
        return sqrt(mean_error)

    def cost(self):
        costValue = 0
        for xi, yi in zip(self.X, self.Y):
            costValue += 0.5 * ((self.hypothesis(xi) - yi)**2)
        return costValue
    
    def derivatives(self):
        dtheta0 = 0
        dtheta1 = 0
        for xi, yi in zip(self.X, self.Y):
            dtheta0 += self.hypothesis(xi) - yi
            dtheta1 += (self.hypothesis(xi) - yi) * xi

        dtheta0 /= self.n
        dtheta1 /= self.n

        return dtheta0, dtheta1
    
    def updateParameters(self):
        dtheta0, dtheta1 = self.derivatives()
        self.theta0 = self.theta0 - (self.lr * dtheta0)
        self.theta1 = self.theta1 - (self.lr * dtheta1)

    def plotLine(self):
        max_x = np.max(self.X) + np.max(self.X) * 0.2
        min_x = np.min(self.X) - np.min(self.X) * 0.2
        max_y = np.max(self.Y) + np.max(self.Y) * 0.2
        min_y = np.min(self.Y) - np.min(self.Y) * 0.2

        xplot = np.linspace(min_x, max_x, 1000)
        yplot = self.theta0 + self.theta1 * xplot

        plt.plot(xplot, yplot, color='#ff0000', label='Regression Line')

        plt.scatter(self.X, self.Y)
        plt.axis([min_x * 1.2, max_x * 1.2, min_y * 1.2, max_y * 1.2])
        plt.show()

    def train(self):
        for i in range(self.num_iter):
            if i % 100 == 0:
                self.plotLine()
            
            self.updateParameters()
                                                              

X, Y = make_regression(n_samples=100, n_features=1, noise=0.4, bias=50)

linear = Linear(X, Y)
linear.train()
print(linear.theta0)
print(linear.theta1)


