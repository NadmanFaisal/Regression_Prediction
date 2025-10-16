import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    def get_gradient_descent(self, alpha, X, w, y):
        m, n = X.shape
        error = (X @ w) - y
        w = w - ((alpha/m) * (X.T @ error))
        return w

    def train(self, X, y, w, alpha, iterations):
        costs = []
        for i in range(iterations):
            w = self.get_gradient_descent(alpha, X, w, y)
            pred_y = self.get_pred_y(X, w)
            cost = self.get_cost_function(pred_y, y)
            costs.append(cost)

        print("Cost:", cost)
        
        # Plot the cost function
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, iterations + 1), costs, marker='o')
        plt.title("Cost Function vs Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.grid(True)
        plt.savefig("Plots/cost_function_vs_iterations.png")
        plt.close()

        return w

    def get_prediction(self, X, w):
        prediction = X @ w
        return prediction

    def get_accuracy(self, prediction, actual):
        ss_res = np.sum((actual - prediction) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
