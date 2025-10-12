import numpy as np
import pandas as pd

from linear_regression import linear_regression
from extract_data import extract_data

extractor = extract_data()

np_array = extractor.convert_to_array('Data/Housing.csv')

y = np_array[:, 0]
X = np_array[:, 1:]
rows, cols = X.shape
X = np.concatenate((np.ones((rows, 1)), X), axis=1)
w = np.zeros(cols + 1)

print("X shape:", X.shape)
print("w shape", w.shape)

model = linear_regression()
model.train(X, y, w)
