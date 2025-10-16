import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_regression import ml_regression

data = pd.read_csv("Data/California_Housing_Data.csv").drop(["total_rooms", "total_bedrooms", "population", "ocean_proximity", "households"], axis=1)

print(data)

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

model = ml_regression()
model.hello_world()
