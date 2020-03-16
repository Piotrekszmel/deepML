import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from math import sqrt
from sklearn.datasets import make_regression
from typing import Union, Tuple


class Linear:
    """
    A linear approach to modeling the relationship 
    between dependent variable and 
    one or more independent variables.

    # Example: 
    
    ```python
        X = [[1,2,3,4,5], [10,11,12,13,14]]
        y = [6, 15]
        
        linear = Linear(X, y, scale=0, verbose=0)
        
        theta, cost_h, theta_h = linear.fit(10000)
        
        predictions = linear.predict([[6,7,8,9,10]])
        
        linear.plotCost(cost_h, linear.num_iter)
    
    @param: X (Union[list, tuple, numpy array]) : independent variables
    @param: y (Union[list, tuple, numpy array]) : dependent variable
    @param: scale (boolean) : if True then scale X variables to mean 0 and stddev 1
    @param: lr (float) : learning rate for updating theta
    @param: verbose (boolean) : if True then plot best fit line (available only for simple Linear Regression)
    
    @hypothesis: return solved linear equation
    @standarize: return scaled X data
    @cost: return cost value for given X and y
    @derivatives: return calculated derivatives for theta
    @fit: return theta, cost history and theta history
    """

    def __init__(self, X: Union[list, tuple, np.array], y: Union[list, tuple, np.array], scale=True, lr=0.005, verbose=0) -> None:
        X = np.array(X)
        self.X = self.standardize(X) if scale == 1 else X
        self.y = y
        self.scale = scale
        self.m = X.shape[1] if X.ndim >= 2 else 1
        self.n = len(y)
        self.lr = lr
        self.verbose = verbose
        self.theta = np.random.randn(X.shape[1] + 1) if X.ndim >= 2 else np.random.randn(2)
    
    def hypothesis(self, X: np.array) -> float:
        if X.ndim == 0:
            X = X.reshape([1,1])
        return self.theta[0] + np.matmul(X, self.theta[1:])

    def standardize(self, X: np.array) -> np.array :
        X = X - np.mean(X) 
        X = X / np.std(X)
        return X
        
    def rmse_metric(self, actual: Union[list, tuple, np.array], predicted: Union[list, tuple, np.array]) -> float:
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

    def fit(self, num_iter: int) -> Tuple[np.array, list, list]:
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
    
    def predict(self, X_test: Union[list, tuple, np.array]) -> list:
        X_test = np.array(X_test)
        self.Y_hat = []
        for xs in X_test:
            self.Y_hat.append(self.hypothesis(xs))
        return self.Y_hat

    def plotLine(self, X: Union[list, tuple, np.array], y: Union[list, tuple, np.array], theta0: np.array, theta: np.array) -> None:
        X = np.array(X)
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

    def plotCost(self, cost_h: list, num_iter: list) -> None:
        plt.plot(list(range(num_iter)), cost_h, "-r")
        plt.show()


class Logistic: 
    def __init__(self, lr, fit_intercept=True, verbose=0):
        self.lr = lr
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z, dtype=np.float128))
    
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def hypothesis(self, X):
        z = np.dot(X, self.theta)
        h = self.sigmoid(z)
        return h

    def gradient(self, X, y, h):
        return np.dot(X.T, (h - y)) / y.shape[0]
    
    def updateParameters(self, X, y, lr, h):
        self.theta -= lr * self.gradient(X, y, h)
    
    def fit(self, X, y, num_iter):
        X = np.array(X)
        y = np.array(y)
        if X.ndim == 1:
            X = X.reshape((X.shape[0], 1))
        
        if self.fit_intercept:
            X = self.add_intercept(X)
        
        self.theta = np.zeros((X.shape[1]))

        for i in range(num_iter):
            h = self.hypothesis(X)
            self.updateParameters(X, y, self.lr, h)

            if self.verbose == 1 and i % 100000 == 0:
                h = self.hypothesis(X)
                print(f"loss: {self.loss(h, y)}")
    
    def predict_probs(self, X, theta):
        return self.sigmoid(np.dot(X, theta))

    def predict(self, X, theta, threshold=0.5):
        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape((X.shape[0], 1))

        if self.fit_intercept:
            X = self.add_intercept(X)

        return self.predict_probs(X, theta) >= threshold




from sklearn.linear_model import LogisticRegression
import sklearn

"""
iris = sklearn.datasets.load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1

logistic = Logistic(lr=0.1, verbose=1)
logistic.fit(X, y, 300000)
preds = logistic.predict(X, logistic.theta)
print("\nMean: ", (preds == y).mean())
print()
model = LogisticRegression(C=1e20)
model.fit(X, y)
preds = model.predict(X)
print("Mean: ", (preds==y).mean())
"""