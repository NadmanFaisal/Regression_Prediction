import numpy as np
import pandas as pd

np.random.seed(42)
X = np.linspace(0, 100, 100)
noise = np.random.normal(0, 10, X.shape)
y = 5 * X + 30 + noise

data = pd.DataFrame({'Feature': X, 'Target': y})
data.to_csv('NoisyLinearData.csv', index=False)
print("✅ Noisy linear data saved as 'Data/NoisyLinearData.csv'")
