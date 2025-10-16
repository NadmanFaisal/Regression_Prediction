import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_regression import ml_regression

data = pd.read_csv("Data/California_Housing_Data.csv").drop(["total_rooms", "total_bedrooms", "population", "ocean_proximity", "households"], axis=1)

## Plot the longitude vs meadian_house_value
#plt.figure()
#plt.scatter(data["longitude"], data["median_house_value"])
#plt.title("Longitude vs median_house_value")
#plt.xlabel("longitude")
#plt.ylabel("median_house_value")
#plt.savefig("Plots/longitude_vs_median_house_value")
#plt.close()
#
## Plot the latitude vs meadian_house_value
#plt.figure()
#plt.scatter(data["latitude"], data["median_house_value"])
#plt.title("latitude vs median_house_value")
#plt.xlabel("latitude")
#plt.ylabel("median_house_value")
#plt.savefig("Plots/latitude_vs_median_house_value")
#plt.close()
#
## Plot the housing_median_age vs meadian_house_value
#plt.figure()
#plt.scatter(data["housing_median_age"], data["median_house_value"])
#plt.title("housing_median_age vs median_house_value")
#plt.xlabel("housing_median_age")
#plt.ylabel("median_house_value")
#plt.savefig("Plots/housing_median_age_vs_median_house_value")
#plt.close()
#
## Plot the median_income vs meadian_house_value
#plt.figure()
#plt.scatter(data["median_income"], data["median_house_value"])
#plt.title("median_income vs median_house_value")
#plt.xlabel("median_income")
#plt.ylabel("median_house_value")
#plt.savefig("Plots/median_income_vs_median_house_value")
#plt.close()

np_data = data.to_numpy()
np.random.seed(42)
np.random.shuffle(np_data)

X_raw = np_data[:, :4].astype(np.float64)
y = np_data[:, -1].astype(np.float64)
m, n = X_raw.shape

# Split data 80/20
split = int(0.8 * m)
X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
y_train, y_test = y[:split], y[split:]

# Scale features
x_mean = X_train_raw.mean(axis=0)
x_min  = X_train_raw.min(axis=0)
x_max  = X_train_raw.max(axis=0)
den = (x_max - x_min)

X_train = (X_train_raw - x_mean) / den
X_test  = (X_test_raw  - x_mean) / den

# Add bias
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test  = np.hstack([np.ones((X_test.shape[0], 1)),  X_test])

# Ensure data
print("X Train:", X_train)

# Train the model
w = np.zeros(X_train.shape[1])
alpha = 0.03
iterations = 6000

model = ml_regression()
w = model.train(X_train, y_train, w, alpha, iterations)

# Check accuracy
prediction = model.get_prediction(X_test, w)
r2 = model.get_accuracy(prediction, y_test)

print("R2:", r2)

plt.figure(figsize=(6,6))
plt.scatter(y_test, X_test @ w, color='green', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Actual vs Predicted")
plt.tight_layout()
plt.savefig("Plots/actual_vs_predicted.png")
plt.close()
