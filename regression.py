import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from math import sqrt
from sklearn.datasets import make_regression

np.random.seed(42)


class Linear:
    def __init__(self, X, y, scale=0, lr=0.005, verbose=0):
        X = np.array(X)
        self.X = self.standardize(X) if scale == 1 else X
        self.y = y
        self.scale = scale
        self.m = X.shape[1] if X.ndim >= 2 else 1
        self.n = len(y)
        self.lr = lr
        self.verbose = verbose
        self.theta = np.random.randn(X.shape[1] + 1) if X.ndim >= 2 else np.random.randn(2)
    
    def hypothesis(self, X):
        if X.ndim == 0:
            X = X.reshape([1,1])
        return self.theta[0] + np.matmul(X, self.theta[1:])

    def standardize(self, X):
        X = X - np.mean(X) 
        X = X / np.std(X)
        return X
        
    def rmse_metric(self, actual, predicted):
        sum_error = 0.0
        for y, y_hat in zip(actual, predicted):
            prediction_error = y_hat - y
            sum_error += (prediction_error ** 2)
        mean_error = sum_error / float(len(actual))
        return sqrt(mean_error)

    def cost(self):
        self.J = 0
        for xs, ys in zip(self.X, self.y):
            h = self.hypothesis(xs)
            self.J += (1 / (2 * self.m)) * np.sum((h - ys)**2)
        return self.J
    
    def derivatives(self):
        dtheta0 = 0
        dtheta = 0
        for xi, yi in zip(self.X, self.y):
            dtheta0 += self.hypothesis(xi) - yi
            dtheta += (self.hypothesis(xi) - yi) * xi

        dtheta0 /= self.n
        dtheta /= self.n

        return dtheta0, dtheta
    
    def updateParameters(self):
        dtheta0, dtheta = self.derivatives()
        self.theta[0] = self.theta[0] - (self.lr / self.m) * dtheta0
        self.theta[1:] = self.theta[1:] - (self.lr / self.m) * dtheta

    def train(self, num_iter):
        self.num_iter = num_iter
        self.cost_history = []
        self.theta_history = []
        for i in range(num_iter):
            self.cost_history.append(self.cost())
            self.theta_history.append(self.theta)
            self.updateParameters()
        if self.verbose == 1:
            self.plotLine(self.X, self.y, self.theta[0], self.theta[1:])                                                  

        return self.theta, self.cost_history, self.theta_history
    
    def predict(self, X_test):
        X_test = np.array(X_test)
        self.Y_hat = []
        for xs in X_test:
            self.Y_hat.append(self.hypothesis(xs))
        return self.Y_hat

    def plotLine(self, X, y, theta0, theta):
        X = np.array(X)
        print(X.shape)        
        assert X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1)
        max_x = np.max(X) + np.max(X) * 0.4
        min_x = np.min(X) - np.min(X) * 0.4
        max_y = np.max(y) + np.max(y) * 0.4
        min_y = np.min(y) - np.min(y) * 0.4

        xplot = np.linspace(min_x, max_x, 5000)
        yplot = theta0 + theta * xplot

        plt.plot(xplot, yplot, color='#ff0000', label='Regression Line')

        plt.scatter(X, y)
        plt.axis([min_x * 1.2, max_x * 1.2, min_y * 1.2, max_y * 1.2])
        plt.show()

    def plotCost(self, cost_h, num_iter):
        plt.plot(list(range(num_iter)), cost_h, "-r")
        plt.show()


X, y = make_regression(n_samples=10, n_features=1, noise=0.4, bias=50)

#X = [[1,2,3,4,5], [10,11,12,13,14]]
#y = [6, 15]
linear = Linear([1,2,3], [4,5,6], scale=0, verbose=1)
theta, cost_h, theta_h = linear.train(10000)

linear.plotCost(cost_h, linear.num_iter)
#predictions = linear.predict([[6,7,8,9,10]])
#print(predictions)





