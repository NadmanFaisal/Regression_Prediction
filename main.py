import os
import numpy as np
import matplotlib.pyplot as plt

from linear_regression import linear_regression
from extract_data import extract_data

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

# Load data
extractor = extract_data()
np_array = extractor.convert_to_array('Data/NoisyLinearData.csv').astype(np.float64)

# First column = feature (X), second column = target (y)
X = np_array[:, 0].reshape(-1, 1)
y = np_array[:, 1]
rows = X.shape[0]

# Plot graph to see relationship
plt.scatter(X, y)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.savefig("plots/test.png")

# Shuffle before splitting
idx = np.random.permutation(rows)
X, y = X[idx], y[idx]

# Feature scaling (min–max using entire dataset, since it’s small)
x_min, x_max = X.min(), X.max()
X_scaled = (X - x_min) / (x_max - x_min)

# Add bias column
X = np.c_[np.ones((rows, 1)), X_scaled]

# Split into train/test sets
split_ratio = 0.8
split_index = int(rows * split_ratio)

X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]

# Train model
model = linear_regression()
w_init = np.zeros(X_train.shape[1])

#m = X_train.shape[0]
#L = (np.linalg.norm(X_train, 2) ** 2) / m
#alpha = 1.0 / (2 * L)
w = model.train(X_train, y_train, np.zeros(X_train.shape[1]), alpha=0.01, iterations=10000)

# Predict & Evaluate
prediction = model.make_prediction(X_test, w)
r2 = model.get_accuracy(prediction, y_test)

print("\nFinal weights:", w)
print(f"R² = {r2:.4f}")

# Plot results
order = np.argsort(X_test[:, 1])
plt.figure()
plt.scatter(X_test[:, 1], y_test, color='blue', label='Actual')
plt.plot(X_test[order, 1], prediction[order], color='red', label='Predicted')
plt.xlabel("Feature (scaled)")
plt.ylabel("Target (y)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/final.png")
plt.close()
