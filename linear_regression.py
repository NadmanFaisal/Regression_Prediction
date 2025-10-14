import numpy as np
import pandas as pd
import csv

class linear_regression:

    def __init__(self):
        pass

    def hello_world(self):
        print("Hello world")

    def get_pred_y(self, X, w):
        pred_y = X @ w
        return pred_y

    def get_cost_function(self, pred_y,  y, X):
        error = pred_y - y
        m, n = X.shape
        cost = (1/(2 * m)) * (np.sum(error * error))
        return cost

    def get_gradient_descent(self, alpha, X, w, y):
        m, n = X.shape
        error = (X @ w) - y
        w = w - ((alpha/m) * (X.T @ error))
        return w

    def train(self, X, y, w, alpha, iterations):
        for i in range(iterations):
            w = self.get_gradient_descent(alpha, X, w, y)
            pred_y = self.get_pred_y(X, w)
            cost = self.get_cost_function(pred_y, y, X)
            print("Cost:", cost)
            print("W:", w)
        
        return w

    def make_prediction(self, X, w):
        prediction = X @ w
        return prediction

    def get_accuracy(self, prediction, actual):
        ss_res = np.sum((actual - prediction) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
