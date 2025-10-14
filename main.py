import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import linear_regression
from extract_data import extract_data

extractor = extract_data()

np_array = extractor.convert_to_array('Data/Housing.csv')

y = np_array[:, 0]
X = np_array[:, 1:]
rows, cols = X.shape

for i in range(cols):
    feature = X[:, i]
    plt.figure()
    plt.scatter(feature, y, alpha=0.6)
    plt.xlabel(f"Feature {i+1}")
    plt.ylabel("Target (y)")
    plt.savefig(f"plots/plot_of_features_{i + 1}.png")
    plt.close()

X = np_array[:, 1].reshape(rows, 1)
X = np.concatenate((np.ones((rows, 1)), X), axis=1)
w = np.zeros(cols + 1)

print("X shape:", X.shape)
print("w shape", w.shape)

#model = linear_regression()
#model.train(X, y, w, alpha=0.01, iterations=25)
