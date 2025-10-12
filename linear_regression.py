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
        print("pred y Shape:", pred_y.shape)
        return pred_y

    def get_cost_function(self, pred_y,  y, X):
        error = y - pred_y
        m, n = X.shape
        cost = (1/(2 * m)) * (np.sum(error * error))
        return cost

    def train(self, X, y, w):
        pred_y = self.get_pred_y(X, w)
        cost = self.get_cost_function(pred_y, y, X)
        print("Cost:", cost)


