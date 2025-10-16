import numpy as np
import pandas as pd

class ml_regression:
    def __init__(self):
        pass

    def hello_world(self):
        print("Hello world")

    def get_pred_y(self, X, w):
        return X @ w

    def get_cost_function(self, pred_y, y):
        error = pred_y - y
        m = len(y)
        cost = (1/(2 * m)) * np.sum(error * error)
        return cost

    def train(self, X, y, w):
        pred_y = self.get_pred_y(X, w)
        cost = self.get_cost_function(pred_y, y)

        print("Cost:", cost)
