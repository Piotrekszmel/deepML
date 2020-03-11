import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from math import sqrt
from sklearn.datasets import make_regression


class Linear:
    def __init__(self, X, y, scale=0, lr=0.005, verbose=0, num_iter=5000):
        self.X = self.standardize(X) if scale == 1 else X
        self.y = y
        self.scale = scale
        self.n = len(y)
        self.lr = lr
        self.verbose = verbose
        self.num_iter = num_iter
        self.theta0 = np.random.rand()
        self.theta1 = np.random.rand()
    
    def hypothesis(self, x):
        return self.theta0 + (self.theta1 * x)

    def standardize(self, X):
        X = X - np.mean(X) 
        X = X / np.std(X)
        return X
        
    def calculate_coeff(self):
        numerator = 0
        denominator = 0
        for x, y in zip(self.X, self.y):
            numerator += (x - self.X_mean) * (y - self.y_mean)
            denominator += (x - self.y_mean) ** 2
        self.theta1 = numerator / denominator
        self.theta0 = self.y_mean - (self.theta1 * self.X_mean)
    
    def rmse_metric(self, actual, predicted):
        sum_error = 0.0
        for y, y_hat in zip(actual, predicted):
            prediction_error = y_hat - y
            sum_error += (prediction_error ** 2)
        mean_error = sum_error / float(len(actual))
        return sqrt(mean_error)

    def cost(self):
        J = 0
        for xi, yi in zip(self.X, self.y):
            J += ((self.hypothesis(xi) - yi)**2) / (2 * self.n)
        return J
    
    def derivatives(self):
        dtheta0 = 0
        dtheta1 = 0
        for xi, yi in zip(self.X, self.y):
            dtheta0 += self.hypothesis(xi) - yi
            dtheta1 += (self.hypothesis(xi) - yi) * xi

        dtheta0 /= self.n
        dtheta1 /= self.n

        return dtheta0, dtheta1
    
    def updateParameters(self):
        dtheta0, dtheta1 = self.derivatives()
        self.theta0 = self.theta0 - (self.lr * dtheta0)
        self.theta1 = self.theta1 - (self.lr * dtheta1)

    def predict(self, X_test):
        if self.scale == 1:
            X_test = self.standardize(X_test)
        predictions = []
        for x in X_test:
            predictions.append(self.hypothesis(x))
        return predictions

    def plotLine(self, X, y, theta0, theta1):
        max_x = np.max(X) + np.max(X) * 0.2
        min_x = np.min(X) - np.min(X) * 0.2
        max_y = np.max(y) + np.max(y) * 0.2
        min_y = np.min(y) - np.min(y) * 0.2

        xplot = np.linspace(min_x, max_x, 1000)
        yplot = theta0 + theta1 * xplot

        plt.plot(xplot, yplot, color='#ff0000', label='Regression Line')

        plt.scatter(X, y)
        plt.axis([min_x * 1.2, max_x * 1.2, min_y * 1.2, max_y * 1.2])
        plt.show()

    def train(self):
        cost_history = [0] * self.num_iter
        for i in range(self.num_iter):
            self.updateParameters()
            cost_history[i] = self.cost()
        if self.verbose == 1:
            self.plotLine(self.X, self.y, self.theta0, self.theta1)                                                  

        return cost_history


class MultipleLinear:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.m = X[0].shape
        self.n = X[1].shape
        self.theta = np.random.randn(X.shape[1])
    
    def hypothesis(self):
        return self.theta[0] + np.matmul(self.X, self.theta[1:])




#X, Y = make_regression(n_samples=100, n_features=1, noise=0.4, bias=50)
X = [1,2,3,4,5]
y = [3,4,5,6,7]
linear = Linear(X, y, scale=0, verbose=1, num_iter=5000)
cost = linear.train()


"""
X_test = [6,700, 8,9000]
Y_test = [8,9,10,11]
predictions = linear.predict(X_test)

print(linear.rmse_metric(Y_test, predictions))

linear = Linear(X, Y, scale=1)
linear.train()

X_test = [6,700,8,9000]
Y_test = [8,9,10,11]


predictions = linear.predict(X_test)
print(linear.rmse_metric(Y_test, predictions))



print("Actual: ", Y_test)
print("Predictions: ", predictions)
"""





